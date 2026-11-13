# PyLBM-Isca

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24084237.svg)](https://doi.org/10.5281/zenodo.24084237)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1PEQdTSNQfhj1O-UJGLtZP0O8mxyxIDK2?usp=sharing)

A **Linear Baroclinic Model** whose dynamical core is a **linearisation of the
Isca / GFDL `atmos_spectral` primitive equations**.

PyLBM-Isca is newly written code that **follows the structure of** the
Watanabe–Kimoto linear baroclinic model (a spectral, sigma-coordinate,
vorticity–divergence, semi-implicit primitive-equation LBM) while deriving its
dynamical core term by term from the **GPLv3** Isca / GFDL `atmos_spectral`
sources. The motivation is **licensing**: the Watanabe–Kimoto Fortran LBM is not
open source — access is by individual permission from the authors — so code
derived from it inherits that restriction. Isca, by contrast, is GPLv3 open
source, and its dynamical core is GFDL's spectral-transform primitive-equation
core. Because PyLBM-Isca shares **no code** with the restricted `ln_solver` and
derives only from the GPLv3 Isca sources and standard published numerical
methods, it can be freely distributed under the GPL.

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

The components that encode the *dynamical core* — the physical constants, the
Simmons–Burridge vertical operators, the linearised tendencies, and the time
filter — are linearisations of the GPLv3 Isca sources. The surrounding numerical
machinery (the spherical-harmonic transforms, the basic-state / forcing / output
pipeline, and the GPU backend) is written from standard published methods and
follows the structure of the Watanabe–Kimoto LBM. It uses the same file and grid
conventions, so the two models are directly comparable and share input and output
formats, but PyLBM-Isca contains no code from the restricted package.

## Provenance (what it is derived from)

Every dynamical term is a linearisation of a specific routine in the GPLv3 Isca
repository `github.com/ExeClim/Isca`:

| PyLBM-Isca | Isca source |
|---|---|
| `constants.py` | `src/shared/constants/constants.F90` (Earth defaults) |
| `vertical.py`  | `src/atmos_spectral/model/press_and_geopot.F90` (`pressure_variables_3d`, `compute_geopotential`) and `spectral_dynamics.F90:four_in_one` |
| `dynamics.py`  | `spectral_dynamics.F90` tendency assembly + `four_in_one`; `vert_advection.F90` (SECOND_CENTERED, ADVECTIVE_FORM) |
| `integrator.py`| `implicit.F90` (per-wavenumber gravity-wave matrix); semi-implicit spectral scheme following Bourke (1974) and Hoskins & Simmons (1975) |
| time filter    | `src/atmos_spectral/model/leapfrog.F90` (Robert–Asselin; Robert 1966, Asselin 1972) |

This is a re-derivation from the cited GPLv3 Isca sources and standard published
numerical methods; it contains **no** code from the permission-restricted
Watanabe–Kimoto `ln_solver`.

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
4. **Time filter** — a genuine Robert–Asselin filter (Robert 1966; Asselin 1972).

## Run in the browser (Google Colab)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1PEQdTSNQfhj1O-UJGLtZP0O8mxyxIDK2?usp=sharing)

The quickest way to try PyLBM-Isca — no local install, and a free CUDA GPU (the
CuPy backend is selected automatically when a GPU runtime is available). The
notebook:

1. clones this repository,
2. runs the **solid-body-rotation** example (stationary Rossby-wave train),
3. downloads the realistic basic state and MJO forcing from Zenodo, and
4. runs the **realistic ERA5 DJF 1979** experiment, including an animated
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
| `bs.1979.bin`   | 11 MB | ERA5 pressure-level basic state (DJF 1979) |
| `frc.phase1.bin`| 46 MB | MJO phase-1 time-varying heating |

`examples/example_3_realistic.ipynb` downloads them into `data/` from Zenodo on
first run (then caches them). To fetch them manually:

```bash
mkdir -p data && cd data
curl -LO "https://zenodo.org/records/24086134/files/bs.1979.bin?download=1"
curl -LO "https://zenodo.org/records/24086134/files/frc.phase1.bin?download=1"
```

From the command line, point `--basic-state` / `--mjo-forcing` at them. The
loaders accept any Fortran-sequential file of the same 17-level layout, so you can
substitute your own basic state / forcing — see [`data/README.md`](data/README.md).

## Quick start

```bash
python run_lbm.py \
    --basic-state data/bs.1979.bin \
    --mjo-forcing data/frc.phase1.bin \
    --days 35 --out out/isca.1979.phase1 --device auto
```

writes `out/isca.1979.phase1.grd` + `.ctl` (GrADS, record order:
`psi, chi, u, v, w, t, z, ps`) and, with `--netcdf`, `.nc`.

```python
from pylbm_isca import Spectral, VerticalGrid, LBM, Config
from pylbm_isca.basicstate import load_basic_state
from pylbm_isca.forcing import MJOForcing
from pylbm_isca.output import GradsWriter

sp, vg = Spectral(), VerticalGrid()
bs = load_basic_state("data/bs.1979.bin", sp, vg)
model = LBM(bs, sp, vg, Config(ndays=35))
model.run(mjo_forcing=MJOForcing("data/frc.phase1.bin"),
          writer=GradsWriter("out/isca.grd", sp, vg))
u200 = model.ga.u[:, :, 12]        # response at sigma ~ 0.175
```

## Configure a run from a file (the namelist)

The Watanabe–Kimoto LBM drives every run from a `SYSIN` file — a stack of
Fortran `&nmXXX … &end` groups that fully specify the integration. PyLBM-Isca
keeps that tradition in a modern, dependency-light form: a single **TOML** (or
**YAML**) file where each table maps one-to-one onto the original namelist
group. Point the driver at it with `--namelist`:

```bash
python run_lbm.py --namelist examples/mjo_phase1.toml
```

```toml
# examples/mjo_phase1.toml  (abridged)
[time]                      # &nmtime / &nmdelt
ndays  = 35
delt   = 1200.0            # time step [s] (= 20 min)

[damp]                      # &nmdamp  — per-level Rayleigh drag, k=0 at bottom
ray0   = 30.0
sponge = true
ddragv = [0.5, 0.5, 0.5, 5, 15, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 0.5, 0.5]

[input]                     # &nminit
basic_state = "data/bs.1979.bin"
[forcing]                   # &nmfrc / &nmsfrc
mjo    = "data/frc.phase1.bin"
spinup = 435
[output]                    # &nmhist / &nmhisd
out    = "out/isca.1979.phase1"
```

| table | Fortran group | knobs |
|---|---|---|
| `[time]`        | `&nmtime` / `&nmdelt` | `ndays`, `delt`, `inistp` |
| `[hdif]`        | `&nmhdif` | `order`, `tefold`, `tunit` (∇⁴ diffusion) |
| `[damp]`        | `&nmdamp` | per-level `ddragv/d/t`, `tunit`, `sponge`, `ray0` |
| `[vdif]`        | `&nmvdif` | per-level `vdifv/d/t/q` |
| `[integration]` | — | `robert`, `strictly_linear`, `nmem` |
| `[input]`       | `&nminit` | `basic_state`, `kmo`, `navg`, `zonal_mean` |
| `[forcing]`     | `&nmfrc` / `&nmsfrc` | `mjo`, `steady`, `case`, `spinup`, `scale`, `south_first` |
| `[output]`      | `&nmhist` / `&nmhisd` | `out`, `netcdf`, `interval_days`, `instantaneous` |
| `[backend]`     | — | `device` |

Any value you omit keeps the model default, and **any command-line flag still
overrides the file** (`--days 10 --device cpu`), so the namelist and the
equivalent command line produce identical runs. Unknown sections or keys raise
immediately, as the Fortran reader did. The file reaches knobs the CLI never
exposed — the per-level `ddrag*`/`vdif*` profiles, `hdif_*`, `ray0`, `nmem`. In
a notebook, build a `Config` straight from the file:

```python
from pylbm_isca import Config
cfg = Config.from_namelist("examples/mjo_phase1.toml")   # integration knobs
```

(YAML needs `pip install pyyaml`; TOML uses the standard-library `tomllib`.)

## Layout

```
pylbm_isca/constants.py   GFDL constants, T42 L20 resolution, sigma half levels
pylbm_isca/spectral.py    spherical-harmonic transforms
pylbm_isca/vertical.py    Simmons-Burridge sigma operators
pylbm_isca/dynamics.py    linearised Isca tendencies
pylbm_isca/integrator.py  semi-implicit leapfrog + damping
pylbm_isca/state.py       grid state, grid<->spectral, Robert-Asselin filter
pylbm_isca/model.py       configuration and the time loop
pylbm_isca/namelist.py    TOML/YAML run-configuration loader (the namelist)
pylbm_isca/basicstate.py  pressure-level -> sigma basic state
pylbm_isca/forcing.py     steady and MJO forcing readers
pylbm_isca/output.py      GrADS / netCDF writers
run_lbm.py              command-line driver
tests/                  validation
```

## Validation

```bash
python tests/test_vertical_isca.py   # Simmons-Burridge coefficients & matrices
python tests/test_physics_isca.py    # resting-state zero tendency; Gill response
python tests/test_spectral.py        # spherical-harmonic transforms
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
3. `example_3_realistic.ipynb` — MJO teleconnection on the ERA5 DJF 1979 state
   (needs the input files above)

You can run examples 2 and 3 with zero setup (and a free GPU) in
[Google Colab](https://colab.research.google.com/drive/1PEQdTSNQfhj1O-UJGLtZP0O8mxyxIDK2?usp=sharing).

## License

GPLv3-or-later — see [`LICENSE`](LICENSE). PyLBM-Isca is a re-derivation from the
GPLv3 Isca sources (`github.com/ExeClim/Isca`); it contains **no** code from the
permission-restricted Watanabe–Kimoto `ln_solver`.

## Citation

If you use PyLBM-Isca, please cite the archived release. The citation metadata
live in [`CITATION.cff`](CITATION.cff); GitHub shows a ready-made citation via
*"Cite this repository"*. The archive is on Zenodo:

- **Concept DOI** (always the latest version): [10.5281/zenodo.24084237](https://doi.org/10.5281/zenodo.24084237)
- **This release (v1.0.0):** [10.5281/zenodo.24114461](https://doi.org/10.5281/zenodo.24114461)

```
Tseng, Kai-Chih (2026). PyLBM-Isca: a Linear Baroclinic Model on the Isca/GFDL
spectral core (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.24114461
```
