# Input data

Binary inputs for the **realistic** experiment
(`examples/example_3_realistic.ipynb`, manuscript Sect. 4.3). The idealised
Gill and solid-body examples build their basic states analytically and need
nothing from this folder.

These files are **not stored in git** — they are archived on Zenodo and
downloaded here on first run:

> **DOI:** [10.5281/zenodo.24086134](https://doi.org/10.5281/zenodo.24086134)
> · record <https://zenodo.org/records/24086134>

| file | size | description |
|---|---|---|
| `bs.1979.bin`    | 11 MB | NCEP pressure-level **basic state**, JFM 1979, Fortran-sequential big-endian `real*4`, ordered month → variable → level (z, rhum, q, t, u, v, omega) on the 128×64 Gaussian grid. Read by `lbm_isca.basicstate.load_basic_state`. |
| `frc.phase1.bin` | 46 MB | **MJO phase-1** time-varying temperature forcing, 73 days × 20 levels on the same grid. Read by `lbm_isca.forcing.MJOForcing`. |

Both files share the layout of the sibling `LBM_python_GPU` package and are
interchangeable with any NCEP file of the same structure — substitute your own
basic state / forcing by pointing `--basic-state` / `--mjo-forcing` (or the
notebook's `DATA` path) elsewhere.
