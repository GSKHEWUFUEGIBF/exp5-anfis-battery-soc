# ANFIS Battery State-of-Charge Estimation

This experiment estimates battery state of charge (SoC) from terminal voltage, load current, and temperature. It compares a linear-regression baseline with first-order Sugeno adaptive neuro-fuzzy inference systems (ANFIS), including a model trained with Z-score-normalized inputs. The repository contains MATLAB and Python implementations plus the supplied dataset and Simulink artifacts.

## Method

- Inputs: voltage (V), current (A), and temperature (°C). Target: SoC (%).
- The 1,000-row dataset is split into 700 training, 150 validation, and 150 held-out test samples. Normalization parameters are calculated from the training split only.
- Candidate grid-partition structures are compared on validation error. The selected model uses `[5, 2, 2]` Gaussian membership functions (20 rules) and hybrid learning: least-squares consequent estimation with gradient-based premise updates.
- A BMS replay applies warning and cut-off flags at 20% and 5% SoC thresholds.
- `python/exp5_anfis_soc.py` implements ANFIS directly with NumPy and produces a separate set of results and plots.

## Held-out results

RMSE is reported in SoC percentage points. MATLAB and Python use separate training implementations, so their results are listed independently.

| Model | MATLAB RMSE | MATLAB R² | Python RMSE | Python R² |
| --- | ---: | ---: | ---: | ---: |
| Linear regression | 4.1384 | 0.98357691 | 4.1384 | 0.98357691 |
| ANFIS, voltage only | 0.1887 | 0.99996587 | 0.1895 | 0.99996558 |
| ANFIS, raw inputs | 0.0252 | 0.99999939 | 0.0059 | 0.99999997 |
| ANFIS, Z-score inputs | 0.00285 | 0.99999999 | 0.00207 | 0.999999996 |

These are results on the included held-out split. The dataset files do not document an external field-data source, so the metrics should be treated as benchmark results for this dataset rather than evidence of real-world battery performance.

## Requirements and run

### MATLAB / Simulink

Requires MATLAB with Fuzzy Logic Toolbox; Simulink is required for the model build and simulation.

From the repository root, run:

```matlab
Exp5_ANFIS_SoC
build_Exp5_Simulink
```

### Python

Requires Python 3, NumPy, and Matplotlib. Install the packages listed in `requirements.txt`, then run:

```console
python python/exp5_anfis_soc.py
```

The Python script looks for `bms_dataset.csv` in its own folder or the repository root and writes plots to `python/figures_python/`.

## Repository contents

- `Exp5_ANFIS_SoC.m`, `anfis_soc.fis`, `anfis_soc_model.mat` — MATLAB training and saved model artifacts.
- `build_Exp5_Simulink.m`, `Exp5_ANFIS_SoC_BMS.slx` — Simulink model builder and model.
- `bms_dataset.csv` — the experiment dataset.
- `matlab_results.json`, `python/python_results.json`, `simulink_results.json` — saved metrics and simulation summary.
- `figures_matlab/`, `figures_simulink/`, `python/figures_python/` — result and model figures.

## Scope

Results depend on the supplied dataset and split. External validation, sensor noise, and hardware tests are outside this experiment. No software license is specified in this repository.
