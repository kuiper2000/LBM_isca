"""The PyLBM-Isca driver: configuration, the time loop and the history.

Semi-implicit leapfrog integration of the linearised Isca/GFDL primitive
equations, with a genuine Robert-Asselin time filter (Isca ``leapfrog.F90``)
(Robert 1966; Asselin 1972).  The namelist-style
defaults in :class:`Config` otherwise follow the same T42 L20 configuration so
the two models are directly comparable.

GPU note
--------
The model inherits its array backend from the :class:`~pylbm_isca.spectral.Spectral`
object passed in (``self.xp``), so the whole time loop runs on whatever device
the transforms live on.  The only host traffic is the daily history.
"""

import time as _time
from dataclasses import dataclass, field

import numpy as np

from .backend import asnumpy, device_name
from .constants import GRAV, KMAX
from .dynamics import BasicStateTerms, Dynamics
from .integrator import Damping, Integrator, SemiImplicit, VerticalDiffusion
from .state import State

# The &nmdamp profile from linearneutral.t42l20.csh, in days, k=0 at the
# bottom: 0.5-day boundary-layer drag in the lowest 3 layers, 0.5-day sponge
# in the top 2, 20-day Newtonian damping in between.
DDRAG_DEFAULT = [0.5, 0.5, 0.5, 5, 15, 20, 20, 20, 20, 20,
                 20, 20, 20, 20, 20, 20, 20, 20, 0.5, 0.5]


@dataclass
class Config:
    """Every namelist knob the run script sets, with the same defaults."""

    # &nmdelt
    delt: float = 1200.0            # s  (delt=20, tunit='MIN')
    inistp: int = 2
    # &nmtime  (start dd=1, end dd=67 -> 66 days of integration)
    ndays: float = 66.0
    # &nmhdif
    hdif_order: int = 4
    hdif_tefold: float = 2.0
    hdif_tunit: str = "HOUR"
    sponge: bool = True
    ray0: float = 30.0
    # &nmdamp  (tunit='DAY')
    ddragv: list = field(default_factory=lambda: list(DDRAG_DEFAULT))
    ddragd: list = field(default_factory=lambda: list(DDRAG_DEFAULT))
    ddragt: list = field(default_factory=lambda: list(DDRAG_DEFAULT))
    ddragq: list = None
    damp_tunit: str = "DAY"
    # &nmvdif  (1.d3 for v, d, t; q unset -> off)
    vdifv: float = 1.0e3
    vdifd: float = 1.0e3
    vdift: float = 1.0e3
    vdifq: float = None
    # Robert-Asselin time-filter coefficient (Isca leapfrog.F90 robert_coeff)
    robert: float = 0.04
    # history
    out_interval_days: float = 1.0
    time_average: bool = True
    # fidelity switches
    strictly_linear: bool = False   # True drops the 0.5*(u'^2+v'^2) KE term
    # ensemble: run this many heating experiments in one batched integration
    nmem: int = 1
    # backend; only used when the driver builds Spectral/VerticalGrid for you
    device: str = "auto"            # "cpu" | "gpu" | "auto"

    @property
    def steps_per_day(self):
        return int(round(86400.0 / self.delt))


class LBM:
    """Linear baroclinic model, T42 L20, dry."""

    def __init__(self, basic_state, spectral, vertical, config=None,
                 basic_spectral=None):
        self.cfg = config or Config()
        self.sp = spectral
        self.vg = vertical
        self.bs = basic_state
        self.xp = spectral.xp

        # Basic-state derived terms; this also builds TBAR and every
        # TBAR-dependent matrix on the vertical grid.  The basic state has no
        # member axis, so when ``spectral`` is a batched (ensemble) transform
        # the setup must use the plain, single-member transform instead.
        self.bst = BasicStateTerms(basic_spectral or self.sp, self.vg,
                                   basic_state)
        self.dyn = Dynamics(self.sp, self.vg, self.bst,
                            strictly_linear=self.cfg.strictly_linear)

        c = self.cfg
        self.damping = Damping(
            self.sp, self.vg, order=c.hdif_order, tefold=c.hdif_tefold,
            tunit_hdif=c.hdif_tunit, ddragv=c.ddragv, ddragd=c.ddragd,
            ddragt=c.ddragt, ddragq=c.ddragq, tunit_damp=c.damp_tunit,
            ray0=c.ray0, sponge=c.sponge)
        self.semi = SemiImplicit(self.sp, self.vg, self.damping)
        self.vdiff = VerticalDiffusion(self.vg, c.vdifv, c.vdifd,
                                       c.vdift, c.vdifq)
        self.integ = Integrator(self.sp, self.vg, self.damping,
                                self.semi, self.vdiff)

        # Use a batched (member-axis) state whenever the transform is batched
        # (so even a single member carries a leading axis of size 1) or when
        # more than one member is requested.
        if getattr(spectral, "batched", False) or self.cfg.nmem > 1:
            nm = max(self.cfg.nmem, 1)
            mk = lambda: State.zeros_ensemble(basic_state.ps, self.vg.kmax, nm,
                                              xp=self.xp)      # noqa: E731
        else:
            mk = lambda: State.zeros(basic_state.ps, self.vg.kmax,
                                     xp=self.xp)                # noqa: E731
        self.ga = mk()          # current level, t
        self.gb = mk()          # previous level, t-dt
        self.gc = mk()          # scratch for the t+dt level (Robert filter)
        self.time = 0.0
        self.istep = 0

    @property
    def device(self):
        return device_name(self.xp)

    # ------------------------------------------------------------------
    def set_initial_perturbation(self, u=None, v=None, t=None, q=None,
                                 ps=None):
        for name, val in (("u", u), ("v", v), ("t", t), ("q", q), ("ps", ps)):
            if val is not None:
                arr = self.xp.asarray(val, dtype=self.xp.float64)
                setattr(self.ga, name, arr.copy())
                setattr(self.gb, name, arr.copy())

    # ------------------------------------------------------------------
    def diagnostics(self):
        """The eight history variables, all in grid space (device arrays).

        Called at the top of the step (before the leapfrog update), so every
        field -- ``psi``, ``chi`` and ``w`` from the tendency computation and
        ``u``, ``v``, ``t``, ``z`` and ``p`` from the state -- is diagnosed
        consistently from the same time level ``t``, using a single top-of-step
        history sample.
        """
        sp, vg = self.sp, self.vg
        d = self.dyn.diag
        psi = sp.wave_to_grid(sp.inverse_laplacian(d["vor_s"]))
        chi = sp.wave_to_grid(sp.inverse_laplacian(d["div_s"]))
        z = vg.hydrostatic(self.ga.t) / GRAV
        return {"psi": psi, "chi": chi, "u": self.ga.u, "v": self.ga.v,
                "w": d["omega"], "t": self.ga.t, "z": z, "p": self.ga.ps}

    # ------------------------------------------------------------------
    def run(self, mjo_forcing=None, steady_forcing=None, writer=None,
            verbose=True, progress_every=240):
        """Integrate forward.

        Parameters
        ----------
        mjo_forcing : MJOForcing or None
            Time-varying temperature forcing.  When present it *replaces* the
            steady temperature forcing.
        steady_forcing : SteadyForcing or None
            Constant vorticity / divergence / temperature / ln(Ps) forcing.
        writer : object with ``.add(day, fields)`` and ``.close()``
        """
        cfg = self.cfg
        xp = self.xp
        spd = cfg.steps_per_day
        nsteps = int(round(cfg.ndays * spd))
        oinit = True
        t0 = _time.time()

        acc = None
        nacc = 0
        out_every = int(round(cfg.out_interval_days * spd))
        day_done = 0
        # History is sampled at the top of the step (all fields at time t, the
        # pre-update state) and trapezoidally
        # averaged over each output interval (mean of this step's and the
        # previous step's samples).  This makes the PyLBM daily history use the
        # same leapfrog sampling point, so the two codes are
        # consistent to round-off in the early response.
        prev_top = None

        # Coerce steady forcing onto the device once (it is time-constant).
        sf = steady_forcing
        sf_vor = xp.asarray(sf.vor) if sf is not None else None
        sf_div = xp.asarray(sf.div) if sf is not None else None
        sf_ps = xp.asarray(sf.ps) if sf is not None else None
        sf_t = xp.asarray(sf.t) if sf is not None else None

        if verbose:
            print(f"  backend: {self.device}", flush=True)

        istep = 0
        advanced = 0
        while advanced < nsteps:
            istep += 1
            if istep > cfg.inistp:
                oinit = False
                dt = cfg.delt
            else:
                dt = cfg.delt / (2.0 ** (cfg.inistp - istep + 1))

            # --- forcing for this call --------------------------------
            frc_t = None
            if mjo_forcing is not None:
                frc_t = xp.asarray(mjo_forcing.step())
            elif sf is not None:
                frc_t = sf_t
            frc_vor = sf_vor
            frc_div = sf_div
            frc_ps = sf_ps

            # --- one step ---------------------------------------------
            tend = self.dyn.tendencies(self.ga, frc_t=frc_t, frc_vor=frc_vor,
                                       frc_div=frc_div, frc_ps=frc_ps)
            # top-of-step diagnostics (all fields at time t, pre-update)
            cur_top = self.diagnostics() if writer is not None else None
            prev = self.gb.to_spectral(self.sp)
            new = self.integ.step(dt, prev, tend)
            self.gc.load_spectral(self.sp, new)          # t+dt in grid space
            if oinit:
                self.gb.assign(self.ga)                  # forward start
            else:
                self.gb.robert_asselin(self.ga, self.gc, cfg.robert)
            self.ga.assign(self.gc)                      # current <- t+dt

            if not self.ga.isfinite():
                raise FloatingPointError(
                    f"model blew up at step {istep} (t={self.time/86400:.2f} d)")

            if oinit:
                prev_top = cur_top
                continue

            self.time += dt
            advanced += 1

            # --- history ----------------------------------------------
            if writer is not None:
                if prev_top is None:
                    prev_top = cur_top
                fields = {k: 0.5 * (cur_top[k] + prev_top[k]) for k in cur_top}
                if cfg.time_average:
                    if acc is None:
                        acc = {k: xp.zeros_like(v) for k, v in fields.items()}
                    for k, v in fields.items():
                        acc[k] += v
                    nacc += 1
                if advanced % out_every == 0:
                    day_done += 1
                    if cfg.time_average:
                        writer.add(day_done,
                                   {k: asnumpy(v / nacc) for k, v in acc.items()})
                        acc = None
                        nacc = 0
                    else:
                        writer.add(day_done,
                                   {k: asnumpy(v) for k, v in fields.items()})
            prev_top = cur_top

            if verbose and progress_every and advanced % progress_every == 0:
                m = self.ga.max_abs()
                el = _time.time() - t0
                print(f"  day {self.time/86400:6.2f}  "
                      f"|u|max {m['u']:7.3f}  |T|max {m['t']:6.3f}  "
                      f"|ps|max {m['ps']:7.4f}   [{el:5.1f}s]", flush=True)

        if verbose:
            print(f"  done: {advanced} steps, {_time.time()-t0:.1f}s")
        if writer is not None:
            writer.close()
        return self
