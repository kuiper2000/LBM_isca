# LBM_isca

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24084237.svg)](https://doi.org/10.5281/zenodo.24084237)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1PEQdTSNQfhj1O-UJGLtZP0O8mxyxIDK2?usp=sharing)

A **Linear Baroclinic Model** whose dynamical core is a **linearisation of the
Isca / GFDL `atmos_spectral` primitive equations**.

This is a clean-room sibling of `LBM_python_GPU` (the GPU port of the
Watanabe–Kimoto `ln_solver`). The motivation is **licensing**: the original
Fortran LBM is not open source — access is by individual permission from the
authors — so any port of it inherits that restriction. Isca, by contrast, is
**GPLv3** open source, and its dynamical core is GFDL's spectral transform
primitive-equation core. By linearising *Isca's* equations instead, LBM_isca
can be freely distributed under the GPL.

| | |
|---|---|
| equations | spectral primitive equations, **pure-sigma** coordinate, linearised about a basic state |
| vertical scheme | **Simmons & Burridge (1981)** energy-conserving differencing (Isca/GFDL) |
| moisture | moist — carries a humidity perturbation `q`, virtual temperature `Tv = T(1+0.608q)` in the pressure-gradient and hydrostatic terms |
| resolution | T42 L20 (128 × 64 Gaussian grid, 20 sigma levels) |
| time scheme | semi-implicit leapfrog + **Robert–Asselin** filter |
| backend | numpy (CPU) **or** CuPy (CUDA GPU), selected at run time |
| physics | **none** — dynamics only (no Isca boundary layer / convection / radiation) |
| licence | **GPLv3** (derived from the GPLv3 Isca sources) |

The only parts that differ from `LBM_python_GPU` are the ones that encode the
*dynamical core*: the physical constants, the Simmons–Burridge vertical
operators, the linearised tendencies, and the time filter. The spherical-
harmonic transforms, the basic-state / forcing / output pipeline and the GPU
backend are the same Python, so the two models are directly comparable and
share input and output formats.

## Provenance (what was ported, and from where)

Every dynamical term is a linearisation of a specific routine in the GPLv3 Isca
repository `github.com/ExeClim/Isca`:

| LBM_isca | Isca source |
|---|---|
| `constants.py` | `src/shared/constants/constants.F90` (Earth defaults) |
| `vertical.py`  | `src/atmos_spectral/model/press_and_geopot.F90` (`pressure_variables_3d`, `compute_geopotential`) and `spectral_dynamics.F90:four_in_one` |
| `dynamics.py`  | `spectral_dynamics.F90` tendency assembly + `four_in_one`; `vert_advection.F90` (SECOND_CENTERED, ADVECTIVE_FORM) |
| `integrator.py`| `implicit.F90` (per-wavenumber gravity-wave matrix) — reused unchanged from the sibling package |
| time filter    | `src/atmos_spectral/model/leapfrog.F90` (Robert–Asselin) |

This is a re-derivation from cited sources, **not** a copy of the proprietary
`ln_solver`. The shared numerical scaffolding (transforms, damping, I/O) is the
project's own existing Python.

## How it differs from the Watanabe–Kimoto LBM

Both are spectral, sigma-coordinate, vorticity–divergence, semi-implicit
primitive-equation models, so the overall structure is the same. The genuine
differences:

1. **Vertical differencing** — Simmons & Burridge (Isca) replaces the
   Arakawa–Suarez coefficients. In pure sigma every S–B coefficient (`alpha`,
   the log-pressure weights, the pressure-gradient weight, the hydrostatic
   matrix) is a ps-independent geometric constant, so it drops straight into the
   same per-wavenumber semi-implicit solver. (In pure sigma the pressure-gradient
   weight `cpg` is exactly 1, so the PGF reduces to the familiar `R·Tv·∇ln ps`.)
2. **Constants** — GFDL values: sidereal `Ω = 7.2921150e-5`, `κ = 2/7`,
   `cp = 1004.64`. (The radius defaults to 6371 km to match the shared grid;
   pass `Spectral(radius=6.376e6)` for Isca's own value.)
3. **Moisture** — virtual temperature in the PGF and geopotential, and a
   prognostic humidity perturbation.
4. **Time filter** — a genuine Robert–Asselin filter rather than the DADVNC shim.

## Run in the browser (Google Colab)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1PEQdTSNQfhj1O-UJGLtZP0O8mxyxIDK2?usp=sharing)

The quickest way to try LBM_isca — no local install, and a free CUDA GPU (the
CuPy backend is selected automatically when a GPU runtime is available). The
notebook:

1. clones this repository,
2. runs the **solid-body-rotation** example (stationary Rossby-wave train),
3. downloads the realistic basic state and MJO forcing from Zenodo, and
4. runs the **realistic NCEP JFM 1979** experiment, including an animated
   upper-level streamfunction response to MJO phase-1 heating.

Open the badge above, then *Runtime → Run all* (pick a GPU runtime under
*Runtime → Change runtime type* for the fastest run).

## Install

```bash
pip install numpy scipy          # required
pip install xarray netCDF4       # optional, for netCDF output
pip install cupy-cuda12x         # optional, for the GPU backend (CUDA 12.x)
```

## Input data

The idealised experiments (Gill, solid-body rotation — see `examples/`) build
their basic states analytically and need **no external files**. The *realistic*
experiment uses two binary inputs that are **archived on Zenodo**
([DOI 10.5281/zenodo.24086134](https://doi.org/10.5281/zenodo.24086134)), not
committed to this repository:

| file | ~size | what it is |
|---|---|---|
| `bs.1979.bin`   | 11 MB | NCEP pressure-level basic state (JFM 1979) |
| `frc.phase1.bin`| 46 MB | MJO phase-1 time-varying heating |

`examples/example_3_realistic.ipynb` downloads them into `data/` from Zenodo on
first run (then caches them). To fetch them manually:

```bash
mkdir -p data && cd data
curl -LO "https://zenodo.org/records/24086134/files/bs.1979.bin?download=1"
curl -LO "https://zenodo.org/records/24086134/files/frc.phase1.bin?download=1"
```

From the command line, point `--basic-state` / `--mjo-forcing` at them. The
loaders accept any Fortran-sequential NCEP file of the same layout, so you can
substitute your own basic state / forcing — see [`data/README.md`](data/README.md).

## Quick start

```bash
python run_lbm.py \
    --basic-state data/bs.1979.bin \
    --mjo-forcing data/frc.phase1.bin \
    --days 35 --out out/isca.1979.phase1 --device auto
```

writes `out/isca.1979.phase1.grd` + `.ctl` (GrADS, gt2gr record order:
`psi, chi, u, v, w, t, z, ps`) and, with `--netcdf`, `.nc`.

```python
from lbm_isca import Spectral, VerticalGrid, LBM, Config
from lbm_isca.basicstate import load_basic_state
from lbm_isca.forcing import MJOForcing
from lbm_isca.output import GradsWriter

sp, vg = Spectral(), VerticalGrid()
bs = load_basic_state("data/bs.1979.bin", sp, vg)
model = LBM(bs, sp, vg, Config(ndays=35))
model.run(mjo_forcing=MJOForcing("data/frc.phase1.bin"),
          writer=GradsWriter("out/isca.grd", sp, vg))
u200 = model.ga.u[:, :, 12]        # response at sigma ~ 0.175
```

## Layout

```
lbm_isca/constants.py   GFDL constants, T42 L20 resolution, sigma half levels
lbm_isca/spectral.py    spherical-harmonic transforms (shared, unchanged)
lbm_isca/vertical.py    Simmons-Burridge sigma operators (new)
lbm_isca/dynamics.py    linearised Isca tendencies (new)
lbm_isca/integrator.py  semi-implicit leapfrog + damping (shared)
lbm_isca/state.py       grid state, grid<->spectral, Robert-Asselin filter
lbm_isca/model.py       configuration and the time loop
lbm_isca/basicstate.py  NCEP p-level -> sigma basic state (shared)
lbm_isca/forcing.py     steady and MJO forcing readers (shared)
lbm_isca/output.py      GrADS / netCDF writers (shared)
run_lbm.py              command-line driver
tests/                  validation
```

## Validation

```bash
python tests/test_vertical_isca.py   # Simmons-Burridge coefficients & matrices
python tests/test_physics_isca.py    # resting-state zero tendency; Gill response
python tests/test_spectral.py        # shared transforms
```

Key checks: a resting isothermal basic state with no forcing gives **exactly
zero** tendencies (no spurious linearisation source terms); the semi-implicit
gravity-wave matrix is well conditioned and its inverse is accurate; equatorial
heating produces a **Gill response** (Kelvin wave east, antisymmetric Rossby
gyre pair west).

## Semi-implicit split (note for maintainers)

The reference gravity-wave matrix is built about the **basic-state mean virtual
temperature** (not Isca's isothermal 300 K `ref_temperature_implicit`), which is
the proper LBM choice. Only the reference *energy-conversion-from-divergence*,
the reference pressure-gradient (`gg`) and the continuity coupling are treated
implicitly; the vertical advection of the reference profile is kept explicit —
a valid, stable split, since the fast gravity waves live in the implicit terms.
The `tmtx` operator is stored with the integrator's sign convention (positive
diagonal); the dynamics carries the full explicit energy conversion and
`div @ tmtx.T` removes the reference part, which the integrator restores
implicitly as `-tmtx @ dbar`.

## Examples

Three runnable notebooks in `examples/` reproduce the model's validation
experiments; each differs only in its **basic state**:

1. `example_1_gill.ipynb` — Gill response on a resting isothermal state
2. `example_2_solidbody.ipynb` — Rossby-wave train on a solid-body-rotation state
3. `example_3_realistic.ipynb` — MJO teleconnection on the NCEP JFM 1979 state
   (needs the input files above)

You can run examples 2 and 3 with zero setup (and a free GPU) in
[Google Colab](https://colab.research.google.com/drive/1PEQdTSNQfhj1O-UJGLtZP0O8mxyxIDK2?usp=sharing).

## License

GPLv3-or-later — see [`LICENSE`](LICENSE). LBM_isca is a re-derivation from the
GPLv3 Isca sources (`github.com/ExeClim/Isca`); it contains **no** code from the
permission-restricted Watanabe–Kimoto `ln_solver`.

## Citation

If you use LBM_isca, please cite the archived release. The citation metadata
live in [`CITATION.cff`](CITATION.cff); GitHub shows a ready-made citation via
*"Cite this repository"*. The archive is on Zenodo:

- **Concept DOI** (always the latest version): [10.5281/zenodo.24084237](https://doi.org/10.5281/zenodo.24084237)
- **This release (v0.1.0):** [10.5281/zenodo.24084238](https://doi.org/10.5281/zenodo.24084238)

```
kuiper2000 (2026). LBM_isca: a Linear Baroclinic Model on the Isca/GFDL
spectral core (v0.1.0). Zenodo. https://doi.org/10.5281/zenodo.24084238
```
