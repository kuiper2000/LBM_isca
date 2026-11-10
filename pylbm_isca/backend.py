"""Array-backend selection for the CPU/GPU builds of the LBM.

The whole model is written against the numpy API.  CuPy mirrors that API on
the GPU almost exactly (same ``fft``, ``matmul``, ``einsum``, ``linalg``), so
the GPU port does not rewrite the maths -- it only swaps the *array module*
every hot-path object carries (``self.xp``) from ``numpy`` to ``cupy`` and
makes sure arrays live on the device.

Design
------
* Objects that own precomputed arrays (``Spectral``, ``VerticalGrid``,
  ``State``, the damping/semi-implicit operators, ...) take a ``device``
  argument, resolve it here with :func:`resolve`, and store the module as
  ``self.xp``.  Their methods then call ``self.xp.something`` instead of
  ``np.something``.
* Free helper functions that receive arrays (e.g. ``_column_sum``) stay
  backend-agnostic by dispatching on their inputs with
  :func:`get_array_module` -- the same idea as ``cupy.get_array_module``, but
  it also works when CuPy is absent.
* One-time setup (reading binary files, the Legendre recurrences, building the
  20x20 vertical matrices) is cheap and stays on the host in numpy; the result
  is moved to the device once with :func:`to_device`.

Selecting the device
--------------------
``device`` may be ``"cpu"``/``"numpy"``, ``"gpu"``/``"cuda"``, or
``"auto"``/``None`` (GPU when a CUDA device is visible, else CPU).  The
environment variable ``LBM_DEVICE`` provides the default for ``"auto"``.
"""

import os

import numpy as np

try:                                   # CuPy is optional -- CPU build needs none
    import cupy as _cp
    try:
        _GPU_OK = _cp.cuda.runtime.getDeviceCount() > 0
    except Exception:                  # pragma: no cover - driver/runtime issue
        _GPU_OK = False
except Exception:                      # pragma: no cover - cupy not installed
    _cp = None
    _GPU_OK = False

cp = _cp


def gpu_available():
    """True when CuPy is importable and at least one CUDA device is visible."""
    return _GPU_OK


def resolve(device):
    """Map a device spec to an array module (``numpy`` or ``cupy``).

    Parameters
    ----------
    device : {"cpu", "numpy", "gpu", "cuda", "auto", None}
        ``"auto"``/``None`` honour ``$LBM_DEVICE`` and otherwise pick the GPU
        when one is present.
    """
    if device is None or str(device).lower() == "auto":
        env = os.environ.get("LBM_DEVICE", "").strip().lower()
        if env:
            return resolve(env)
        return cp if _GPU_OK else np

    d = str(device).lower()
    if d in ("gpu", "cuda"):
        if not _GPU_OK:
            raise RuntimeError(
                "device='gpu' requested but CuPy/CUDA is not available. "
                "Install cupy-cuda12x and check `nvidia-smi`.")
        return cp
    if d in ("cpu", "numpy"):
        return np
    raise ValueError(f"unknown device {device!r}")


def get_array_module(*arrays):
    """Return the module (``numpy`` or ``cupy``) that owns ``arrays``.

    Backend-agnostic helpers use this to work on either device without
    carrying an explicit ``xp``.
    """
    if cp is not None:
        return cp.get_array_module(*arrays)
    return np


def asnumpy(x):
    """Bring an array to the host as a numpy array (no-op for numpy input)."""
    if cp is not None:
        return cp.asnumpy(x)
    return np.asarray(x)


def to_device(x, xp):
    """Move ``x`` onto the device backing ``xp`` (host->GPU, or no-op)."""
    return xp.asarray(x)


def device_name(xp):
    """Human-readable name for an array module."""
    return "gpu" if (cp is not None and xp is cp) else "cpu"


def synchronize(xp=None):
    """Block until queued GPU work finishes -- needed for honest timing.

    A no-op on the CPU backend.
    """
    if cp is not None and (xp is None or xp is cp):
        cp.cuda.Stream.null.synchronize()
