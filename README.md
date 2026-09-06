# Solar Activity and Synoptic-Map Prediction

Research code accompanying the article **“Prediction of solar activities:
Sunspot numbers and solar magnetic synoptic maps”**. The repository predicts
sunspot numbers and the large-scale photospheric magnetic field through
spherical-harmonic time series, Long Short-Term Memory (LSTM) networks, and
Empirical Mode Decomposition (EMD).

> This repository preserves the numerical methods used in the study while
> replacing machine-specific paths with a portable configuration layer.

## Associated publication

Zhuo, R., He, J., Duan, D. *et al.* Prediction of solar activities: Sunspot
numbers and solar magnetic synoptic maps. *Science China Earth Sciences* **67**,
2460–2477 (2024).

- [Publisher page](https://link.springer.com/article/10.1007/s11430-023-1354-4)
- [DOI](https://doi.org/10.1007/s11430-023-1354-4)
- [BibTeX and publication notes](docs/PAPER.md)
- Machine-readable citation: [`CITATION.cff`](CITATION.cff)

## Scientific workflow

1. Preprocess WSO photospheric-field maps, spherical-harmonic coefficients,
   and SILSO sunspot-number records.
2. Diagnose temporal variability with continuous wavelet analysis.
3. Forecast sunspot numbers and spherical-harmonic coefficients using LSTM,
   EMD–LSTM, STL–LSTM, SARIMA, FFT, and comparison models.
4. Reconstruct photospheric synoptic maps from the predicted coefficients.
5. Propagate the maps with PFSS/SWMF workflows and evaluate correlations and
   magnetic-flux contributions.

## Repository layout

```text
config/                    Shared JSON and MATLAB configuration
src/smp/                   Python configuration, constants, and utilities
preprocess/                WSO and sunspot-number preprocessing
time_series_analysis/      Harmonic-coefficient diagnostics and CWT
predict_sunspot_number/    Solar-cycle prediction
predict_synoptic_map/      LSTM/EMD and baseline map forecasts
plot_synoptic_map/         Spherical-harmonic map reconstruction and validation
postprocess_on_class/      PFSS/SWMF postprocessing
contribution_analysis/     Active-region magnetic-flux analysis
determine_order/           Spherical-harmonic truncation comparison
tests/                     Lightweight configuration and utility tests
```

`FDIP`, `FDIP_SWMF`, and `PSI-master` are intentionally outside the current
refactoring scope. If present in another checkout, they should be treated as
external/legacy components.

## Installation

Python 3.10 or 3.11 is recommended. The frozen environment used for
reproducibility is in `requirements.txt`; `pyproject.toml` provides compatible
version ranges for development.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

MATLAB scripts require a release with `jsondecode` (R2016b or newer). Before
running them, execute:

```matlab
run('setup_matlab.m')
```

## Configuration and data

The committed defaults use repository-local `data/` and `outputs/` directories.
Copy `config/local.example.json` to `config/local.json` and adjust the two roots.
`config/local.json` is ignored by Git. The environment variables
`SMP_DATA_ROOT`, `SMP_OUTPUT_ROOT`, and `SMP_RANDOM_SEED` take precedence.
Expected input names are documented in [`docs/DATA.md`](docs/DATA.md).

## Running the workflows

Run scripts from the repository root after installing the package, for example:

```bash
python predict_sunspot_number/LSTM_for_sn_sm.py
python predict_synoptic_map/EMD/EMD_LSTM.py
python time_series_analysis/get_harmonic_coefficient_cwt.py
python postprocess_on_class/pfss.py
```

The common random seed is `456` unless overridden. Deep-learning training can
produce slightly different floating-point results across CPU/GPU architectures.
The `original/` transformer scripts and files named “副本” are retained only for
provenance and are not the recommended entry points.

## Verification

```bash
python -m compileall -q src autoencoder_decoder postprocess_on_class \
  predict_sunspot_number predict_synoptic_map time_series_analysis
python -m unittest discover -s tests
```

Full scientific regression requires the WSO, GONG, SILSO, and OMNI inputs.

## Data and publication rights

The original observations retain the terms of their respective providers. The
publisher-formatted article PDF is not redistributed because it does not carry
an open-content licence; use the DOI link above. No software licence has yet
been selected; until a `LICENSE` file is added, reuse rights remain reserved.
