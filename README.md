```markdown
# PDAC Decision-Stability Certificates

This repository contains the code and preprocessed data used for the computational illustration in the manuscript:

**Information-Projection Certificates for Bayes-Decision Stability: Exact, Pairwise and Distribution-Free Bounds**

The analysis demonstrates how information-projection stability diagnostics can be computed from probabilistic classifier outputs in a sparse binary decision-support workflow.

The dataset is a preprocessed version of the publicly released pancreatic cancer biomarker data from Debernardi et al. The manuscript uses two probabilistic classifiers:

1. An artificial neural network, implemented as a one-hidden-layer `MLPClassifier`.
2. XGBoost, implemented as `XGBClassifier`.

## Overview

The script `generate_pdac_stability_figures.py` runs a progressive Monte Carlo repeated-holdout experiment. For each repeated split, models are trained on nested training prefixes and evaluated on a fixed validation set.

Step-to-step changes in predicted probabilities are converted into Bernoulli KL displacement quantities:

```text
IG_step = KL(Bern(p_curr) || Bern(p_prev))
IG_min  = 2 * |p_prev - tau|^2
```

where `tau = 0.5` is the decision threshold.

If a threshold decision flips between successive training sizes, the deterministic Pinsker-based certificate implies:

```text
IG_step >= IG_min
```

The empirical quantities in this repository measure predictive-distribution displacement between successive fitted classifiers. They should not be interpreted as formal Bayesian posterior-to-prior information gain.

## Repository Contents

```text
.
├── PDAC.csv
├── generate_pdac_stability_figures.py
└── README.md
```

## Input Data

### `PDAC.csv`

`PDAC.csv` is the preprocessed PDAC biomarker dataset used in the manuscript.

Expected format:

- The final column is the binary target unless `TARGET_COL` is set in the script.
- The target must be encoded as `0/1` or be convertible to `0/1`.
- Feature columns should be numeric.
- Missing values are dropped by the script.

## Outputs

Running the script creates an output directory:

```text
outputs/<run_id>/
```

where `<run_id>` is a timestamp generated at runtime.

The main output files are:

```text
outputs/<run_id>/progressive_mc_results_summary.csv
outputs/<run_id>/theorem_points_long.csv
outputs/<run_id>/tables/aggregated_stability_tables.csv
outputs/<run_id>/tables/overall_stability_summary.csv
outputs/<run_id>/figures/
```

### Output Descriptions

| File or directory | Description |
|---|---|
| `progressive_mc_results_summary.csv` | Repetition-level predictive and stability summaries. |
| `theorem_points_long.csv` | Patient-level step-to-step stability diagnostics. |
| `tables/aggregated_stability_tables.csv` | Aggregated values used for manuscript tables and plots. |
| `tables/overall_stability_summary.csv` | Overall flip counts and tightness summaries. |
| `figures/` | Manuscript-ready figures with filenames matching the LaTeX source. |

## Manuscript Figures

The script generates manuscript-ready figures with the filenames used in the LaTeX manuscript:

```text
fig_theorem_ann.png
fig_theorem_xgb.png
fig_flip_rate_ann.png
fig_flip_rate_xgb.png
fig_ig_ann.png
fig_ig_xgb.png
fig_auroc_ann.png
fig_auroc_xgb.png
fig_auprc_ann.png
fig_auprc_xgb.png
fig_roc_ann.png
fig_roc_xgb.png
fig_precision_ann.png
fig_precision_xgb.png
fig_recall_ann.png
fig_recall_xgb.png
fig_specificity_ann.png
fig_specificity_xgb.png
fig_f1_ann.png
fig_f1_xgb.png
fig_brier_ann.png
fig_brier_xgb.png
```

These files are saved in:

```text
outputs/<run_id>/figures/
```

## Reproducing the Figures

### 1. Install Dependencies

Install the required Python packages:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

### 2. Place the Data File

Ensure that `PDAC.csv` is in the same directory as:

```text
generate_pdac_stability_figures.py
```

### 3. Run the Script

```bash
python generate_pdac_stability_figures.py
```

### 4. Locate the Outputs

After the script completes, the manuscript-ready figures will be saved under:

```text
outputs/<run_id>/figures/
```

The generated CSV files will be saved under:

```text
outputs/<run_id>/
outputs/<run_id>/tables/
```

## Reproducibility Notes

- The global seed is fixed at `42`.
- Each Monte Carlo repetition uses the repetition index as the random state.
- Feature scaling is fitted on the training pool only and then applied to the validation set.
- The same fitted scaler is used for all nested training prefixes within a repetition.
- Predicted probabilities are clipped to `[1e-12, 1 - 1e-12]` before KL calculations.
- Inequality checks use numerical tolerance `1e-10`.
- The manuscript reports only the ANN/MLP and XGBoost models.

## Methodological Notes

The workflow uses progressive training sizes:

```text
100, 110, 120, 130, 140
```

For each Monte Carlo repetition:

1. The cohort is split into a training pool and validation set.
2. A scaler is fitted on the training pool only.
3. Nested training prefixes are constructed from the training pool.
4. ANN and XGBoost models are trained at each prefix size.
5. Predicted probabilities are evaluated on the same validation set.
6. Step-to-step Bernoulli KL displacement is computed for each validation patient.
7. Threshold flips and stability-certificate quantities are recorded.

## Interpretation

The key empirical quantities are:

| Quantity | Meaning |
|---|---|
| `IG_step` | Bernoulli KL displacement between successive predicted probabilities. |
| `IG_min` | Pinsker threshold certificate for a flip. |
| `rho = IG_step / IG_min` | Margin-normalised displacement ratio. |
| `flip` | Indicator that the threshold decision changed between successive training sizes. |
| `violates` | Numerical implementation check for the deterministic flip inequality. |

The `violates` variable is not a statistical test of the theorem. For flipped Bernoulli probabilities, the inequality is deterministic up to numerical tolerance.

## Data Source

The data derive from the publicly released dataset associated with:

> Debernardi et al., pancreatic cancer biomarker study.

Users should cite the original Debernardi et al. study when using the dataset.

## License

Please add the appropriate license for this repository before public release.

Suggested options:

- MIT License for code.
- A separate data-use note for `PDAC.csv`, depending on the terms of the original dataset.

## Citation

If you use this repository, please cite the manuscript:

```text
Christakis, N., and Drikakis, D.
Information-Projection Certificates for Bayes-Decision Stability:
Exact, Pairwise and Distribution-Free Bounds.
```
```
