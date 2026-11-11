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
| `bs.1979.bin`    | 11 MB | **ERA5** reanalysis **basic state**, DJF 1979, interpolated onto the model's 17 standard pressure levels. Fortran-sequential big-endian `real*4`, ordered month → variable → level (z, rhum, q, t, u, v, omega) on the 128×64 Gaussian grid. Read by `pylbm_isca.basicstate.load_basic_state`. |
| `bs.jja.bin`     | 11 MB | **ERA5** reanalysis **summer (JJA) basic state**, same layout as `bs.1979.bin`, used for the seasonality comparison (`analysis/summer_validation.py`). The fields were downloaded only up to 100 hPa, so the levels above are extrapolated; the analysis holds them isothermally above 100 hPa (those levels lie in the top sponge). Supply a full-depth file to avoid this. |
| `frc.phase1.bin` | 46 MB | **MJO phase-1** time-varying temperature forcing, 73 days × 20 levels on the same grid. Read by `pylbm_isca.forcing.MJOForcing`. |

Both files use a Fortran-sequential, big-endian `real*4` layout and are
interchangeable with any file of the same structure — substitute your own
basic state / forcing by pointing `--basic-state` / `--mjo-forcing` (or the
notebook's `DATA` path) elsewhere.

## Provenance

`bs.1979.bin` and `frc.phase1.bin` are the author's own derived products: the
basic state is computed from **ERA5** reanalysis (DJF 1979) by the preprocessing
in `pylbm_isca.basicstate`, and the MJO phase-1 heating is constructed as
described in the manuscript (Sect. 4.3). Neither file is taken from the
permission-restricted Watanabe–Kimoto LBM distribution.
