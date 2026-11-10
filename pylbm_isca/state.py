"""Perturbation state in grid space, and the grid <-> spectral conversions.

The Fortran carries the perturbation in *grid* space at two time levels
(``GA`` at t, ``GB`` at t-dt) and converts to spectral coefficients inside
every step.  We mirror that layout so the time filter and
the forcing both act where the reference LBM applies them.

Surface pressure deserves a note.  The prognostic spectral variable is
``pi' = ln(ps_bar + ps') - ln(ps_bar)``, i.e. the *exact* logarithm, and the
inverse is ``ps' = ps_bar*(exp(pi') - 1)``.  The
usual linearisation ``pi' ~ ps'/ps_bar`` is not what the code does, so we
keep the exact round trip.

GPU note
--------
All five fields live on the device (``self.xp``).  Every operation here is an
elementwise array op or a transform call, so nothing needs the host; the
constructor moves any host input onto the device once.
"""

import numpy as np

from .backend import asnumpy, resolve


class State:
    """Perturbation fields on the Gaussian grid."""

    __slots__ = ("xp", "u", "v", "t", "q", "ps", "ps_bar", "lnps_bar")

    def __init__(self, ps_bar, u=None, v=None, t=None, q=None, ps=None,
                 device="auto", xp=None):
        self.xp = xp if xp is not None else resolve(device)
        xp = self.xp
        self.ps_bar = xp.asarray(ps_bar, dtype=xp.float64)
        self.lnps_bar = xp.log(self.ps_bar)
        shape3 = None
        for f in (u, v, t, q):
            if f is not None:
                shape3 = f.shape
                break
        if shape3 is None:
            raise ValueError("need at least one 3-D field to infer the shape")
        z3 = lambda: xp.zeros(shape3)                       # noqa: E731
        self.u = z3() if u is None else xp.asarray(u, dtype=xp.float64)
        self.v = z3() if v is None else xp.asarray(v, dtype=xp.float64)
        self.t = z3() if t is None else xp.asarray(t, dtype=xp.float64)
        self.q = z3() if q is None else xp.asarray(q, dtype=xp.float64)
        self.ps = (xp.zeros(self.ps_bar.shape) if ps is None
                   else xp.asarray(ps, dtype=xp.float64))

    # ------------------------------------------------------------------
    @classmethod
    def zeros(cls, ps_bar, nlev, device="auto", xp=None):
        nlat, nlon = ps_bar.shape
        xp_ = xp if xp is not None else resolve(device)
        return cls(ps_bar, u=xp_.zeros((nlat, nlon, nlev)), xp=xp_)

    @classmethod
    def zeros_ensemble(cls, ps_bar, nlev, nmem, device="auto", xp=None):
        """A batched state carrying ``nmem`` members on a leading axis.

        The 3-D fields become ``(nmem, nlat, nlon, nlev)`` and surface pressure
        ``(nmem, nlat, nlon)``; ``ps_bar`` stays 2-D and broadcasts over
        members.  Every per-step operation then runs on all members at once.
        """
        nlat, nlon = ps_bar.shape
        xp_ = xp if xp is not None else resolve(device)
        obj = cls(ps_bar, u=xp_.zeros((nmem, nlat, nlon, nlev)), xp=xp_)
        obj.ps = xp_.zeros((nmem, nlat, nlon))
        return obj

    def copy(self):
        return State(self.ps_bar, self.u.copy(), self.v.copy(),
                     self.t.copy(), self.q.copy(), self.ps.copy(), xp=self.xp)

    def lnps_pert(self):
        """pi' = ln(ps_bar + ps') - ln(ps_bar)."""
        return self.xp.log(self.ps_bar + self.ps) - self.lnps_bar

    def set_from_lnps(self, pi):
        """ps' = ps_bar*(exp(pi') - 1)."""
        self.ps = self.ps_bar * (self.xp.exp(pi) - 1.0)

    # ------------------------------------------------------------------
    def to_spectral(self, sp):
        vor, div = sp.uv_to_vordiv(self.u, self.v)
        return {"vor": vor,
                "div": div,
                "t": sp.grid_to_wave(self.t),
                "pi": sp.grid_to_wave(self.lnps_pert()),
                "q": sp.grid_to_wave(self.q)}

    def load_spectral(self, sp, spec):
        self.u, self.v = sp.vordiv_to_uv(spec["vor"], spec["div"])
        self.t = sp.wave_to_grid(spec["t"])
        self.q = sp.wave_to_grid(spec["q"])
        self.set_from_lnps(sp.wave_to_grid(spec["pi"]))
        return self

    def assign(self, other):
        """Copy every field of ``other`` into this state (in place)."""
        self.u = other.u.copy()
        self.v = other.v.copy()
        self.t = other.t.copy()
        self.q = other.q.copy()
        self.ps = other.ps.copy()
        return self

    def robert_asselin(self, current, new, gamma=0.04):
        """Robert-Asselin time filter (Isca leapfrog.F90:217-247).

        ``self`` holds the *previous* (t-dt) level on entry; it is overwritten
        with the filtered *current* level

            X_f(t) = X(t) + gamma * [ X(t-dt) - 2 X(t) + X(t+dt) ]

        so that, after the caller sets ``current <- new``, ``self`` is the
        filtered previous level for the next leapfrog step.  This is the genuine
        Asselin filter (Robert 1966; Asselin 1972): it sees the t+dt level
        ``new``.
        """
        for name in ("u", "v", "t", "q", "ps"):
            prev = getattr(self, name)
            cur = getattr(current, name)
            nxt = getattr(new, name)
            setattr(self, name, cur + gamma * (prev - 2.0 * cur + nxt))
        return self

    # ------------------------------------------------------------------
    def max_abs(self):
        xp = self.xp
        return {"u": float(xp.abs(self.u).max()),
                "v": float(xp.abs(self.v).max()),
                "t": float(xp.abs(self.t).max()),
                "ps": float(xp.abs(self.ps).max()),
                "q": float(xp.abs(self.q).max())}

    def isfinite(self):
        xp = self.xp
        return all(bool(xp.all(xp.isfinite(x)))
                   for x in (self.u, self.v, self.t, self.q, self.ps))

    def to_numpy(self):
        """Host copies of the five fields, for inspection or output."""
        return {"u": asnumpy(self.u), "v": asnumpy(self.v),
                "t": asnumpy(self.t), "q": asnumpy(self.q),
                "ps": asnumpy(self.ps)}
