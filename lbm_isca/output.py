"""History output: GrADS flat binary (drop-in for ``gt2gr``) and netCDF.

``gt2gr`` (solver/util/gt2gr.f) writes one Fortran sequential record per
level, big-endian ``real*4``, in the order

    psi(20) chi(20) u(20) v(20) w(20) t(20) z(20) ps(1)  = 141 records/time

on the model's north-first grid, which is why the companion ``.ctl`` carries
``OPTIONS SEQUENTIAL YREV``.  Two details of ``gt2gr`` are reproduced here:

* ``opl=f`` in SETPAR, so there is **no** sigma -> pressure interpolation and
  the output stays on the 20 sigma levels.  The sample ``.ctl`` files that
  list 1000, 950, ... hPa assume ``opl=t`` and do not apply to this run.
* ``Z`` gets a surface-pressure term added (gt2gr.f:396-397):
  ``Z_out = Z' + Ps'[hPa]*100/(rho0*g)`` with ``rho0=1.225``, ``g=9.80``.
"""

import os

import numpy as np

from .backend import asnumpy
from .constants import GRAV_GT2GR, RHO00

VARS = ["psi", "chi", "u", "v", "w", "t", "z", "p"]
VAR_INFO = {
    "psi": ("stream function", "m**2/s"),
    "chi": ("velocity potential", "m**2/s"),
    "u": ("zonal wind", "m/s"),
    "v": ("meridional wind", "m/s"),
    "w": ("p-vertical velocity", "hPa/s"),
    "t": ("temperature", "K"),
    "z": ("geopotential height", "m"),
    "p": ("surface pressure", "hPa"),
}


class GradsWriter:
    """Writes ``out.grd`` + ``out.ctl`` byte-compatible with ``gt2gr``."""

    def __init__(self, path, spectral, vertical, ctl_path=None,
                 apply_gt2gr_z=True, title="LBM T42L20 dry (python)"):
        self.path = path
        self.ctl_path = ctl_path or (os.path.splitext(path)[0] + ".ctl")
        self.sp, self.vg = spectral, vertical
        self.apply_gt2gr_z = apply_gt2gr_z
        self.title = title
        self.ntime = 0
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.fh = open(path, "wb")

    def _record(self, a2d):
        """One Fortran sequential record of a (nlat, nlon) real*4 field."""
        data = np.ascontiguousarray(a2d, dtype=">f4")
        marker = np.array([data.size * 4], dtype=">i4")
        marker.tofile(self.fh)
        data.tofile(self.fh)
        marker.tofile(self.fh)

    def add(self, day, fields):
        f = {k: asnumpy(v) for k, v in fields.items()}
        if self.apply_gt2gr_z:
            f["z"] = f["z"] + f["p"][..., None] * 1.0e2 / (RHO00 * GRAV_GT2GR)
        for name in VARS:
            a = f[name]
            if a.ndim == 2:
                self._record(a)
            else:
                for k in range(a.shape[-1]):
                    self._record(a[..., k])
        self.ntime += 1

    def close(self):
        if not self.fh.closed:
            self.fh.close()
        self.write_ctl()

    def write_ctl(self):
        sp, vg = self.sp, self.vg
        lat = sp.lat_deg[::-1]            # ctl lists south -> north with YREV
        lines = [
            f"DSET ^{os.path.basename(self.path)}",
            "OPTIONS SEQUENTIAL YREV BIG_ENDIAN",
            f"TITLE {self.title}",
            "UNDEF -999.",
            f"XDEF {sp.nlon} LINEAR 0. {360.0/sp.nlon:.6g}",
            f"YDEF {sp.nlat} LEVELS",
        ]
        for i in range(0, len(lat), 6):
            lines.append(" " + " ".join(f"{x:9.4f}" for x in lat[i:i + 6]))
        lines.append(f"ZDEF {vg.kmax} LEVELS")
        for i in range(0, vg.kmax, 6):
            lines.append(" " + " ".join(f"{x:.6g}" for x in vg.sig[i:i + 6]))
        lines.append(f"TDEF {max(self.ntime,1)} LINEAR 01jan0001 1dy")
        lines.append(f"VARS {len(VARS)}")
        for name in VARS:
            desc, unit = VAR_INFO[name]
            nz = 0 if name == "p" else vg.kmax
            lines.append(f"{name:4s} {nz:3d} 99 {desc} [{unit}]")
        lines.append("ENDVARS")
        with open(self.ctl_path, "w") as fh:
            fh.write("\n".join(lines) + "\n")


class NetCDFWriter:
    """Optional netCDF output (requires netCDF4 or xarray)."""

    def __init__(self, path, spectral, vertical, apply_gt2gr_z=True):
        self.path = path
        self.sp, self.vg = spectral, vertical
        self.apply_gt2gr_z = apply_gt2gr_z
        self.days = []
        self.data = {v: [] for v in VARS}

    def add(self, day, fields):
        f = {k: asnumpy(v) for k, v in fields.items()}
        if self.apply_gt2gr_z:
            f["z"] = f["z"] + f["p"][..., None] * 1.0e2 / (RHO00 * GRAV_GT2GR)
        self.days.append(day)
        for v in VARS:
            a = f[v]
            # store single precision, matching the real*4 of the GrADS path -
            # float64 would double the file for no extra information
            self.data[v].append(
                np.moveaxis(a, -1, 0).astype(np.float32)
                if a.ndim == 3 else a.astype(np.float32))

    def close(self):
        try:
            import xarray as xr
        except ImportError as exc:                      # pragma: no cover
            raise RuntimeError(
                "netCDF output needs xarray; use GradsWriter instead") from exc
        sp, vg = self.sp, self.vg
        coords = {"time": np.array(self.days, dtype="f8"),
                  "sigma": vg.sig,
                  "lat": sp.lat_deg,
                  "lon": sp.lon_deg}
        ds = xr.Dataset(coords=coords)
        for v in VARS:
            arr = np.array(self.data[v])
            dims = (("time", "lat", "lon") if arr.ndim == 3
                    else ("time", "sigma", "lat", "lon"))
            desc, unit = VAR_INFO[v]
            ds[v] = (dims, arr, {"long_name": desc, "units": unit})
        ds.attrs["title"] = "LBM T42L20 dry, python port"
        ds.attrs["note"] = ("latitudes are the T42 Gaussian grid, north first;"
                            " vertical coordinate is sigma")
        ds.to_netcdf(self.path)
        return ds


class MultiWriter:
    """Fan a history stream out to several writers."""

    def __init__(self, *writers):
        self.writers = [w for w in writers if w is not None]

    def add(self, day, fields):
        for w in self.writers:
            w.add(day, fields)

    def close(self):
        for w in self.writers:
            w.close()
