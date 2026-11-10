"""Ensemble / batch mode: run N heating experiments in one integration.

The linear baroclinic model is linear in the perturbation and shares one basic
state, one set of gravity-wave matrices, and one set of Legendre tables across
all experiments.  The only thing that differs between members is the forcing
(and hence the evolving state).  So an ensemble of N heating experiments is a
single integration with a *member* axis prepended to the state.

Because the GPU is far from saturated at T42 (the per-step cost is launch- and
dispatch-bound, not arithmetic-bound; see the performance section), batching N
members costs little more than one until the device fills up.  Throughput --
experiments per unit time -- therefore rises almost linearly with the ensemble
size until saturation, a capability a serial CPU implementation cannot offer.

Design
------
* ``BatchedSpectral`` subclasses :class:`~lbm.spectral.Spectral` and makes the
  horizontal transforms member-aware by *folding* the leading member axis into
  the transform's column (batch) dimension and unfolding on the way out.  The
  single-member transforms are reused verbatim, so the per-column arithmetic --
  and thus the numerics -- are unchanged.
* ``dynamics.py`` and ``vertical.py`` already broadcast over a leading axis
  (basic-state fields and per-level vectors broadcast against ``(E, ...)``), and
  the semi-implicit / vertical-diffusion contractions in ``integrator.py`` use
  an ellipsis, so nothing else needs to change.
* The basic state has no member axis, so it is built with a plain
  ``Spectral`` (passed to :class:`~lbm.model.LBM` as ``basic_spectral``).

See ``tests/test_gpu.py`` / ``paper/benchmark_ensemble.py`` for the parity
check (E identical members reproduce the single-member run to round-off; E
distinct members reproduce E separate runs) and the throughput curve.
"""

import numpy as np

from .model import LBM, Config
from .spectral import Spectral
from .vertical import VerticalGrid


class BatchedSpectral(Spectral):
    """A :class:`Spectral` whose transforms carry a leading member axis.

    Every public array that reaches this object in ensemble mode has shape
    ``(E, ...)``: grid volumes ``(E, nlat, nlon, k)``, grid/spectral surfaces
    ``(E, A, B)``, spectral volumes ``(E, m, n, k)``.  Each transform folds the
    member axis into the column dimension, calls the inherited single-member
    transform, and unfolds -- so members are processed in the *same* kernel
    launches but with identical per-member numerics.
    """

    batched = True

    # -- fold the leading member axis into the transform column dimension ----
    def _fold(self, x):
        x = self.xp.asarray(x)
        E = x.shape[0]
        if x.ndim == 3:                       # (E, A, B) -> (A, B, E)
            return self.xp.moveaxis(x, 0, -1), (E, None)
        if x.ndim == 4:                       # (E, A, B, k) -> (A, B, E*k)
            k = x.shape[-1]
            y = self.xp.moveaxis(x, 0, -2)    # (A, B, E, k)
            return y.reshape(y.shape[:-2] + (E * k,)), (E, k)
        raise ValueError(f"BatchedSpectral expects 3-D or 4-D input, got "
                         f"{x.ndim}-D")

    def _unfold(self, y, meta):
        E, k = meta
        if k is None:                         # (A, B, E) -> (E, A, B)
            return self.xp.moveaxis(y, -1, 0)
        z = y.reshape(y.shape[:-1] + (E, k))  # (A, B, E, k)
        return self.xp.moveaxis(z, -2, 0)     # (E, A, B, k)

    # -- leaf transforms (the rest dispatch through these via ``self``) ------
    def grid_to_wave(self, f):
        f, meta = self._fold(f)
        return self._unfold(super().grid_to_wave(f), meta)

    def wave_to_grid(self, xi):
        xi, meta = self._fold(xi)
        return self._unfold(super().wave_to_grid(xi), meta)

    def laplacian(self, xi):
        xi, meta = self._fold(xi)
        return self._unfold(super().laplacian(xi), meta)

    def inverse_laplacian(self, xi):
        xi, meta = self._fold(xi)
        return self._unfold(super().inverse_laplacian(xi), meta)

    def hgrad(self, xi):
        xi, meta = self._fold(xi)
        gx, gy = super().hgrad(xi)
        return self._unfold(gx, meta), self._unfold(gy, meta)

    def uv_to_vordiv(self, u, v):
        u, meta = self._fold(u)
        v, _ = self._fold(v)
        vor, div = super().uv_to_vordiv(u, v)
        return self._unfold(vor, meta), self._unfold(div, meta)

    def psichi_to_uv(self, psi, chi):
        psi, meta = self._fold(psi)
        chi, _ = self._fold(chi)
        u, v = super().psichi_to_uv(psi, chi)
        return self._unfold(u, meta), self._unfold(v, meta)


class EnsembleForcing:
    """A constant, per-member temperature forcing presented as an MJO forcing.

    ``step()`` returns the same ``(E, nlat, nlon, kmax)`` device array every
    call, so the model's time loop treats it exactly like a steady thermal
    forcing but with one heating field per ensemble member.
    """

    def __init__(self, gtf, xp):
        self.gtf = xp.asarray(gtf)            # (E, nlat, nlon, kmax), K/s

    def step(self):
        return self.gtf


def build_heating_sweep(base_heat, nmem, mode="longitude", xp=np):
    """Make ``nmem`` heating fields from a single base field.

    Parameters
    ----------
    base_heat : (nlat, nlon, kmax) array
        One heating field (K/s), e.g. the MJO phase-1 day-0 heating.
    mode : {"longitude", "amplitude"}
        ``"longitude"`` rolls the heating to ``nmem`` equally spaced longitudes
        (heating at many locations); ``"amplitude"`` scales it over a range.

    Returns ``(nmem, nlat, nlon, kmax)``.
    """
    base = np.asarray(base_heat)
    nlat, nlon, kmax = base.shape
    out = np.empty((nmem, nlat, nlon, kmax), dtype=np.float64)
    if mode == "longitude":
        for e in range(nmem):
            shift = int(round(e * nlon / nmem))
            out[e] = np.roll(base, shift, axis=1)
    elif mode == "amplitude":
        for e, s in enumerate(np.linspace(0.5, 1.5, nmem)):
            out[e] = s * base
    else:
        raise ValueError(mode)
    return xp.asarray(out)


def make_ensemble_model(basic_state, nmem, config=None, device="gpu"):
    """Build an ensemble :class:`LBM` with ``nmem`` members on one device.

    Returns ``(model, batched_spectral, vertical_grid)``.  The basic state is
    set up with a plain single-member transform; the time loop uses the batched
    transform.
    """
    sp_plain = Spectral(device=device)
    sp_batch = BatchedSpectral(device=device)
    vg = VerticalGrid(device=device)
    cfg = config or Config()
    cfg.nmem = nmem
    model = LBM(basic_state, sp_batch, vg, cfg, basic_spectral=sp_plain)
    return model, sp_batch, vg
