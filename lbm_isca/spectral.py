"""Spherical-harmonic transforms on the T42 Gaussian grid.

This reproduces the conventions of the CCSR/NIES spectral core used by the
LBM (model/src/util/uspst.F, usphe.F, model/src/dynamics/dvect.F):

* Fully normalised associated Legendre functions,
  ``(1/2) * int_{-1}^{1} [P_n^m(mu)]^2 dmu = 1``  (no Condon-Shortley phase).
* Gaussian weights normalised so that ``sum_j gw_j = 1``, giving the
  discrete orthogonality ``sum_j gw_j P_n^m P_n'^m = delta_{nn'}``.
* Latitudes ordered **north to south**: ``j=0`` is +87.8638 deg.  This is the
  model's internal order (uspst.F:291) and the reason every sample GrADS
  ``.ctl`` in the package carries ``OPTIONS YREV``.

GPU note
--------
Every transform is an FFT along longitude followed by a batched matrix
multiply against the Legendre tables (``pnm``/``hnm``).  Both map directly onto
GPU primitives, so the only change from the CPU port is that the tables and the
operators live on the device (``self.xp`` is ``cupy``) and the public methods
accept host *or* device input, moving it onto the device on entry.  The
Legendre recurrences themselves are a one-time host-side setup.

Field layout
------------
Grid fields are ``(nlat, nlon)`` or ``(nlat, nlon, nk)``.
Spectral fields are ``(nmax+1, nmax+1)`` or ``(nmax+1, nmax+1, nk)``, indexed
``[m, n]``; entries with ``n < m`` are identically zero (triangular
truncation).  Coefficients are complex, with the expansion

    f(lam, mu) = sum_m sum_n xi[m,n] P_n^m(mu) exp(i m lam)

where negative-m coefficients are the conjugates of the positive ones, so a
real grid field is reconstructed as ``xi[0] + 2*Re(sum_{m>=1} ...)``.
"""

import numpy as np

from .backend import resolve, to_device
from .constants import ER, IMAX, JMAX, NMAX


class Spectral:
    """Gaussian grid + spherical-harmonic transform operators."""

    #: True for the ensemble (member-axis) subclass; see lbm.ensemble.
    batched = False

    def __init__(self, nlon=IMAX, nlat=JMAX, nmax=NMAX, radius=ER,
                 device="auto"):
        self.xp = resolve(device)
        self.nlon = int(nlon)
        self.nlat = int(nlat)
        self.nmax = int(nmax)
        self.a = float(radius)

        # --- Gaussian latitudes, ordered north -> south (host setup) -------
        mu, w = np.polynomial.legendre.leggauss(self.nlat)
        order = np.argsort(-mu)              # descending mu == north first
        self.mu = mu[order]                  # host; used for Coriolis & output
        self.gw = w[order] / 2.0             # so that sum(gw) == 1
        self.lat = np.arcsin(self.mu)        # radians (host)
        coslat = np.sqrt(1.0 - self.mu ** 2)
        self.lon = np.arange(self.nlon) * (2.0 * np.pi / self.nlon)

        # Weight carrying the 1/(a(1-mu^2)) metric factor; this is GWDEL in
        # dsphe.F:150 and is what makes the vorticity/divergence analysis work.
        self.gwdel = self.gw / self.a / (1.0 - self.mu ** 2)

        self._build_legendre()             # builds host tables, moves to device

        # Laplacian eigenvalues  -n(n+1)/a^2  (dsetc.F:116-122)
        n = np.arange(self.nmax + 1, dtype=np.float64)
        nn1 = n * (n + 1.0)
        self.nn1 = nn1                                         # host
        laplace = -nn1 / self.a ** 2                          # shape (nmax+1,)
        with np.errstate(divide="ignore", invalid="ignore"):
            inv_laplace = np.where(nn1 > 0, -self.a ** 2 / nn1, 0.0)

        # Device-resident operators used inside the hot loop.
        self.coslat = to_device(coslat, self.xp)
        self.laplace = to_device(laplace, self.xp)
        self.inv_laplace = to_device(inv_laplace, self.xp)
        self.im = to_device(1j * np.arange(self.nmax + 1, dtype=np.float64),
                            self.xp)

        # Triangular mask: valid where n >= m (host; diagnostics only)
        mm, nn = np.meshgrid(np.arange(self.nmax + 1),
                             np.arange(self.nmax + 1), indexing="ij")
        self.mask = (nn >= mm)

    # ------------------------------------------------------------------
    def _build_legendre(self):
        """Fully normalised P_n^m and H_n^m = (1-mu^2) dP_n^m/dmu.

        Recurrences follow LGNDRE/LGNDRD (uspst.F:141-237).  P is built one
        degree beyond the truncation because H_n^m needs P_{n+1}^m.  This is a
        one-time host (numpy) computation; the resulting tables are moved to
        the device at the end.
        """
        nmax, nlat, mu = self.nmax, self.nlat, self.mu
        nx = nmax + 2                        # need up to degree nmax+1
        p = np.zeros((nmax + 1, nx, nlat))   # [m, n, j]

        sin_t = np.sqrt(1.0 - mu ** 2)       # cos(latitude)

        # eps(n,m) = sqrt((n^2-m^2)/(4n^2-1))
        n_arr = np.arange(nx, dtype=np.float64)
        eps = np.zeros((nmax + 1, nx))
        for m in range(nmax + 1):
            nn = n_arr[m + 1:]
            eps[m, m + 1:] = np.sqrt((nn ** 2 - m ** 2) / (4.0 * nn ** 2 - 1.0))
        self.eps = eps

        # Sectoral: P_0^0 = 1, P_m^m = P_{m-1}^{m-1} * sqrt((2m+1)/(2m)) * sin_t
        p[0, 0] = 1.0
        for m in range(1, nmax + 1):
            p[m, m] = p[m - 1, m - 1] * np.sqrt((2.0 * m + 1.0) / (2.0 * m)) * sin_t
        # First off-sectoral: P_{m+1}^m = P_m^m * sqrt(2m+3) * mu
        for m in range(nmax + 1):
            if m + 1 < nx:
                p[m, m + 1] = p[m, m] * np.sqrt(2.0 * m + 3.0) * mu
        # General recurrence
        for m in range(nmax + 1):
            for n in range(m + 2, nx):
                p[m, n] = (p[m, n - 1] * mu
                           - p[m, n - 2] * eps[m, n - 1]) / eps[m, n]

        # H_n^m = (1-mu^2) dP_n^m/dmu
        #       = -n*eps_{n+1}^m P_{n+1}^m + (n+1)*eps_n^m P_{n-1}^m
        h = np.zeros((nmax + 1, nmax + 1, nlat))
        for m in range(nmax + 1):
            for n in range(m, nmax + 1):
                term = -n * eps[m, n + 1] * p[m, n + 1] if n + 1 < nx else 0.0
                if n - 1 >= m:
                    term = term + (n + 1.0) * eps[m, n] * p[m, n - 1]
                h[m, n] = term

        pnm = np.ascontiguousarray(p[:, :nmax + 1, :])        # [m, n, j]
        hnm = np.ascontiguousarray(h)

        # Pre-weighted analysis operators (fold the quadrature weight in once).
        # Everything the hot loop touches is moved onto the device; the
        # transposed synthesis tables are cached so we never transpose per call.
        xp = self.xp
        self.pnm = to_device(pnm, xp)                         # [m, n, j]
        self.hnm = to_device(hnm, xp)
        self.pnm_gw = to_device(pnm * self.gw, xp)            # [m, n, j]
        self.pnm_gwdel = to_device(pnm * self.gwdel, xp)
        self.hnm_gwdel = to_device(hnm * self.gwdel, xp)
        # Synthesis uses P^T / H^T (swap the n and j axes): cache once.
        self.pnm_T = to_device(np.swapaxes(pnm, 1, 2), xp)    # [m, j, n]
        self.hnm_T = to_device(np.swapaxes(hnm, 1, 2), xp)

    # ------------------------------------------------------------------
    # Fourier helpers
    # ------------------------------------------------------------------
    def _fft_fwd(self, f):
        """Grid -> Fourier.  f:(nlat,nlon,nk) -> (m, j, k) complex."""
        xp = self.xp
        fm = xp.fft.rfft(f, axis=1) / self.nlon
        fm = fm[:, :self.nmax + 1]
        return xp.ascontiguousarray(xp.moveaxis(fm, 0, 1))     # (m, j, k)

    def _fft_inv(self, fm, nk):
        """Fourier -> grid.  fm:(m, j, k) -> (nlat, nlon, nk)."""
        xp = self.xp
        full = xp.zeros((self.nlat, self.nlon // 2 + 1, nk),
                        dtype=xp.complex128)
        full[:, :self.nmax + 1] = xp.moveaxis(fm, 0, 1)
        return xp.fft.irfft(full * self.nlon, n=self.nlon, axis=1)

    def _as3d(self, f):
        """Coerce host/device input to a device 3-D array; flag 2-D input."""
        f = self.xp.asarray(f)
        return (f[..., None], True) if f.ndim == 2 else (f, False)

    @staticmethod
    def _unpack(x, squeeze):
        return x[..., 0] if squeeze else x

    # ------------------------------------------------------------------
    # Scalar transforms
    # ------------------------------------------------------------------
    def grid_to_wave(self, f):
        """Grid -> spectral.  xi[m,n,k] = sum_j gw_j P_n^m(mu_j) F_m(mu_j)."""
        f3, sq = self._as3d(f)
        fm = self._fft_fwd(f3)                       # (m, j, k)
        xi = self.xp.matmul(self.pnm_gw, fm)         # (m, n, k)
        return self._unpack(xi, sq)

    def wave_to_grid(self, xi):
        """Spectral -> grid."""
        xi3, sq = self._as3d(xi)
        fm = self.xp.matmul(self.pnm_T, xi3)         # (m, j, k)
        return self._unpack(self._fft_inv(fm, xi3.shape[-1]), sq)

    # ------------------------------------------------------------------
    # Differential operators
    # ------------------------------------------------------------------
    def laplacian(self, xi):
        return xi * self.laplace[None, :, ...][..., None] if xi.ndim == 3 \
            else xi * self.laplace[None, :]

    def inverse_laplacian(self, xi):
        f = self.inv_laplace[None, :]
        return xi * f[..., None] if xi.ndim == 3 else xi * f

    def hgrad(self, xi):
        """Physical horizontal gradient of a scalar (HGRAD, dvect.F:205-246).

        Returns ``(dX/dx, dX/dy)`` in grid space, where
        ``dX/dx = 1/(a cos(phi)) dX/dlam`` and ``dX/dy = (1/a) dX/dphi``.
        """
        xp = self.xp
        xi3, sq = self._as3d(xi)
        nk = xi3.shape[-1]
        imxi = self.im[:, None, None] * xi3
        fx = xp.matmul(self.pnm_T, imxi)
        fy = xp.matmul(self.hnm_T, xi3)
        scale = 1.0 / (self.a * self.coslat)[:, None, None]
        gx = self._fft_inv(fx, nk) * scale
        gy = self._fft_inv(fy, nk) * scale
        return self._unpack(gx, sq), self._unpack(gy, sq)

    def uv_to_vordiv(self, u, v):
        """Physical (u, v) -> spectral vorticity and divergence (UV2VDW).

        zeta = 1/(a(1-mu^2)) dV/dlam - (1/a) dU/dmu,  U = u cos(phi)
        div  = 1/(a(1-mu^2)) dU/dlam + (1/a) dV/dmu,  V = v cos(phi)
        """
        xp = self.xp
        u3, sq = self._as3d(u)
        v3, _ = self._as3d(v)
        cos3 = self.coslat[:, None, None]
        um = self._fft_fwd(u3 * cos3)
        vm = self._fft_fwd(v3 * cos3)
        imu = self.im[:, None, None] * um
        imv = self.im[:, None, None] * vm
        pg, hg = self.pnm_gwdel, self.hnm_gwdel
        vor = xp.matmul(pg, imv) + xp.matmul(hg, um)
        div = xp.matmul(pg, imu) - xp.matmul(hg, vm)
        return self._unpack(vor, sq), self._unpack(div, sq)

    def neg_divergence(self, fx, fy):
        """-div(F) for a physical vector field (HCONVW, dvect.F:167-201)."""
        vor, div = self.uv_to_vordiv(fx, fy)
        return -div

    def curl(self, fx, fy):
        """k . curl(F) for a physical vector field, in spectral space."""
        vor, div = self.uv_to_vordiv(fx, fy)
        return vor

    def vordiv_to_uv(self, vor, div):
        """Spectral vorticity/divergence -> physical (u, v) (VD2UVG)."""
        psi = self.inverse_laplacian(vor)
        chi = self.inverse_laplacian(div)
        return self.psichi_to_uv(psi, chi)

    def psichi_to_uv(self, psi, chi):
        """Streamfunction/velocity potential -> physical (u, v)."""
        xp = self.xp
        psi3, sq = self._as3d(psi)
        chi3, _ = self._as3d(chi)
        nk = psi3.shape[-1]
        pt = self.pnm_T
        ht = self.hnm_T
        imp = self.im[:, None, None]
        um = xp.matmul(pt, imp * chi3) - xp.matmul(ht, psi3)
        vm = xp.matmul(pt, imp * psi3) + xp.matmul(ht, chi3)
        scale = 1.0 / (self.a * self.coslat)[:, None, None]
        u = self._fft_inv(um, nk) * scale
        v = self._fft_inv(vm, nk) * scale
        return self._unpack(u, sq), self._unpack(v, sq)

    # ------------------------------------------------------------------
    def truncate(self, f):
        """Round-trip a grid field through the truncation."""
        return self.wave_to_grid(self.grid_to_wave(f))

    @property
    def lat_deg(self):
        return np.degrees(self.lat)

    @property
    def lon_deg(self):
        return np.degrees(self.lon)
