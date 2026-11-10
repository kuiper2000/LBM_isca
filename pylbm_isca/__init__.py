"""PyLBM-Isca - a Linear Baroclinic Model on the Isca/GFDL spectral core.

A GPU-capable linearisation of the Isca (``atmos_spectral``) primitive
equations about a basic state: spectral primitive equations in pure-sigma
coordinates with Simmons & Burridge energy-conserving vertical differencing,
semi-implicit leapfrog, Robert-Asselin filtering.  Dynamics only -- no Isca
physics parameterisations.  Open source (GPLv3): it follows the structure of the Watanabe-Kimoto linear
baroclinic model but shares no code with the restricted ``ln_solver``.

Quick start
-----------
>>> from pylbm_isca import Spectral, VerticalGrid, LBM, Config
>>> from pylbm_isca.basicstate import load_basic_state
>>> from pylbm_isca.forcing import MJOForcing
>>> sp, vg = Spectral(), VerticalGrid()
>>> bs = load_basic_state("bs.1979.bin", sp, vg)
>>> model = LBM(bs, sp, vg, Config())
>>> model.run(mjo_forcing=MJOForcing("frc.phase1.bin"))
"""

from .backend import asnumpy, device_name, gpu_available, resolve
from .constants import IMAX, JMAX, KMAX, NMAX
from .model import LBM, Config
from .spectral import Spectral
from .state import State
from .vertical import VerticalGrid

__all__ = ["Spectral", "VerticalGrid", "LBM", "Config", "State",
           "IMAX", "JMAX", "KMAX", "NMAX",
           "gpu_available", "device_name", "asnumpy", "resolve"]
__version__ = "0.1.0-isca"
