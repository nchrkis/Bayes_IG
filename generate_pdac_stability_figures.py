"""
generate_pdac_stability_figures.py

Bayes-Decision Stability


Authors:
    Nicholas Christakis
    

Repository purpose:
    This script reproduces the computational illustration and manuscript
    figures for the paper on Bayes-Decision Stability by Christakis and Drikakis

    The script uses the preprocessed PDAC.csv dataset derived from the
    publicly released Debernardi et al. pancreatic cancer biomarker data.
    It runs a progressive Monte Carlo repeated-holdout experiment using
    two probabilistic classifiers reported in the manuscript:

        1. ANN:
           Implemented as a one-hidden-layer scikit-learn MLPClassifier.

        2. XGBoost:
           Implemented as an XGBClassifier.

    The aim is not to develop or validate a clinical prediction model.
    The aim is to demonstrate how information-projection stability
    diagnostics can be computed from probabilistic classifier outputs in
    a sparse decision-support workflow.

Main computational idea:
    For each Monte Carlo repetition, the dataset is split into a training
    pool and a held-out validation set. Within the training pool, nested
    training prefixes of increasing size are used:

        n in {100, 110, 120, 130, 140}

    For a fixed validation patient x, the script compares probabilities
    from consecutive training sizes:

        p_prev(x) = predicted probability after training on n samples
        p_curr(x) = predicted probability after training on n + 10 samples

    It computes the Bernoulli KL displacement

        IG_step(x) = KL(Bern(p_curr(x)) || Bern(p_prev(x)))

    and the Pinsker threshold certificate

        IG_min(x; tau) = 2 * |p_prev(x) - tau|^2,

    where tau = 0.5 is the decision threshold.

    If the threshold decision flips between p_prev and p_curr, then
    the deterministic inequality is implied:

        IG_step(x) >= IG_min(x; tau).

    Boundary satisfaction is therefore an implementation check. The
    substantive empirical quantities are flip incidence, stepwise
    Bernoulli KL displacement, and the margin-normalised ratio

        rho(x) = IG_step(x) / IG_min(x; tau).

Input:
    PDAC.csv

    Requirements:
        - The final column is the binary target unless TARGET_COL is set.
        - The target must be encoded as 0/1 or be convertible to 0/1.
        - Feature columns are assumed to be numeric in the preprocessed file.
        - Missing values are dropped.

Outputs:
    All outputs are written to:

        outputs/<run_id>/

    Main CSV files:
        progressive_mc_results_summary.csv
        theorem_points_long.csv
        aggregated_stability_tables.csv
        overall_stability_summary.csv

    Manuscript figures:
        figures/fig_theorem_ann.png
        figures/fig_theorem_xgb.png
        figures/fig_flip_rate_ann.png
        figures/fig_flip_rate_xgb.png
        figures/fig_ig_ann.png
        figures/fig_ig_xgb.png
        figures/fig_auroc_ann.png
        figures/fig_auroc_xgb.png
        figures/fig_auprc_ann.png
        figures/fig_auprc_xgb.png
        figures/fig_roc_ann.png
        figures/fig_roc_xgb.png
        figures/fig_precision_ann.png
        figures/fig_precision_xgb.png
        figures/fig_recall_ann.png
        figures/fig_recall_xgb.png
        figures/fig_specificity_ann.png
        figures/fig_specificity_xgb.png
        figures/fig_f1_ann.png
        figures/fig_f1_xgb.png
        figures/fig_brier_ann.png
        figures/fig_brier_xgb.png

Reproducibility notes:
    - RANDOM_SEED_GLOBAL = 42 is used for the initial global shuffle.
    - The repetition index is used as the random state for the split,
      progressive ordering, ANN initialisation and XGBoost training.
    - MinMaxScaler is fitted on the training pool only within each
      repetition and then applied to the validation set.
    - The same fitted scaler is used for all nested prefixes within a
      repetition. Thus validation leakage is avoided, although the
      preprocessing is not a fully sequential prefix-by-prefix pipeline.
    - Predicted probabilities are clipped to [1e-12, 1 - 1e-12] before
      KL calculations.
    - Inequality checks use tolerance 1e-10.

Usage:
    python generate_pdac_stability_figures.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
)


# =============================================================================
# SETTINGS
# =============================================================================

RANDOM_SEED_GLOBAL = 42

# Decision threshold used throughout the manuscript.
TAU = 0.5

# Monte Carlo repetitions.
N_REPETITIONS = 20

# Progressive training sizes.
INITIAL_TRAIN_SIZE = 100
STEP_SIZE = 10

# Validation fraction per repetition.
VAL_SIZE = 0.30

# Numerical stability.
EPS = 1e-12
VIOL_TOL = 1e-10

# If the target is not the final column, set TARGET_COL to its column name.
TARGET_COL = None

# Input data file.
DATA_FILE = "PDAC.csv"

# Plot style.
sns.set_style("white")
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 600,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
})


# =============================================================================
# OUTPUT DIRECTORIES
# =============================================================================

RUN_ID = time.strftime("%Y%m%d_%H%M%S")
BASE_DIR = Path("outputs") / RUN_ID
FIGURES_DIR = BASE_DIR / "figures"
TABLES_DIR = BASE_DIR / "tables"

for directory in [BASE_DIR, FIGURES_DIR, TABLES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def coerce_binary_target(y: pd.Series) -> np.ndarray:
    """
    Convert a target column to a numpy array of binary 0/1 labels.
    """
    y_raw = y.copy()

    if y_raw.dtype == bool:
        return y_raw.astype(int).to_numpy()

    if np.issubdtype(y_raw.dtype, np.number):
        y_num = pd.to_numeric(y_raw, errors="coerce")
        if y_num.isna().any():
            raise ValueError("Target contains NaNs after numeric coercion.")
        unique_values = set(np.unique(y_num))
        if not unique_values.issubset({0, 1}):
            raise ValueError(
                f"Target numeric values are not binary 0/1: "
                f"{sorted(list(unique_values))[:10]}"
            )
        return y_num.astype(int).to_numpy()

    y_str = y_raw.astype(str).str.strip().str.lower()
    mapping = {
        "0": 0,
        "1": 1,
        "no": 0,
        "yes": 1,
        "false": 0,
        "true": 1,
        "neg": 0,
        "pos": 1,
        "negative": 0,
        "positive": 1,
        "nocancer": 0,
        "no-cancer": 0,
        "no cancer": 0,
        "cancer": 1,
        "pdac": 1,
    }

    if not set(y_str.unique()).issubset(set(mapping.keys())):
        raise ValueError(
            f"Unrecognized target labels: "
            f"{sorted(list(set(y_str.unique())))[:20]}"
        )

    return y_str.map(mapping).astype(int).to_numpy()


def bern_kl(p: np.ndarray, q: np.ndarray, eps: float = EPS) -> np.ndarray:
    """
    Compute KL(Bern(p) || Bern(q)) for vector-valued p and q.
    """
    p = np.clip(p.astype(float), eps, 1.0 - eps)
    q = np.clip(q.astype(float), eps, 1.0 - eps)
    return p * np.log(p / q) + (1.0 - p) * np.log((1.0 - p) / (1.0 - q))


def stratified_progressive_order(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Create a single class-aware ordering of training-pool indices.

    The ordering interleaves shuffled class-0 and class-1 indices so that
    early prefixes stay close to the full training-pool class proportion.
    """
    y = y.astype(int)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]

    rng.shuffle(idx0)
    rng.shuffle(idx1)

    n = len(y)
    p1 = len(idx1) / max(n, 1)

    order = []
    i0 = 0
    i1 = 0

    for k in range(n):
        desired_ones = int(round(p1 * (k + 1)))
        take_one = (i1 < desired_ones) and (i1 < len(idx1))

        if i0 >= len(idx0):
            take_one = True
        if i1 >= len(idx1):
            take_one = False

        if take_one:
            order.append(idx1[i1])
            i1 += 1
        else:
            order.append(idx0[i0])
            i0 += 1

    return np.asarray(order, dtype=int)


def build_models(random_state: int) -> dict[str, object]:
    """
    Build the two probabilistic classifiers reported in the manuscript.

    Model key "ann" corresponds to the one-hidden-layer MLPClassifier.
    Model key "xgb" corresponds to XGBoost.
    """
    return {
        "ann": MLPClassifier(
            hidden_layer_sizes=(20,),
            activation="relu",
            solver="adam",
            max_iter=2000,
            learning_rate_init=0.01,
            alpha=1e-4,
            random_state=random_state,
            early_stopping=True,
            n_iter_no_change=30,
            validation_fraction=0.15,
        ),
        "xgb": XGBClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.02,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=-1,
            eval_metric="logloss",
        ),
    }


def compute_metrics_from_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict[str, float]:
    """
    Compute threshold-based and threshold-free predictive metrics.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_proba))
        if len(np.unique(y_true)) == 2 else np.nan,
        "auprc": float(average_precision_score(y_true, y_proba))
        if len(np.unique(y_true)) == 2 else np.nan,
        "brier": float(brier_score_loss(y_true, y_proba)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def plot_mean_std(
    x: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    ylabel: str,
    title: str,
    outpath: Path,
) -> None:
    """
    Save a mean +/- standard deviation line plot.
    """
    mask = np.isfinite(x) & np.isfinite(y_mean)

    x = x[mask]
    y_mean = y_mean[mask]
    y_std = y_std[mask]

    plt.figure(figsize=(7.2, 5.4))
    plt.plot(x, y_mean, linewidth=2.2)
    plt.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.25)
    plt.xlabel("Training size")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()


def aggregate_curves(
    curves: list[tuple[np.ndarray, np.ndarray]],
    grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Interpolate curves to a common grid and return mean and standard deviation.
    """
    interpolated = []

    for x, y in curves:
        x = np.asarray(x)
        y = np.asarray(y)
        order = np.argsort(x)
        x_sorted = x[order]
        y_sorted = y[order]
        y_interp = np.interp(grid, x_sorted, y_sorted)
        interpolated.append(y_interp)

    values = np.vstack(interpolated)
    return values.mean(axis=0), values.std(axis=0)


def save_final_roc_figure(
    curves: list[tuple[np.ndarray, np.ndarray]],
    model_label: str,
    outpath: Path,
) -> None:
    """
    Save the final ROC curve figure 
    """
    fpr_grid = np.linspace(0.0, 1.0, 201)
    tpr_mean, tpr_std = aggregate_curves(curves, fpr_grid)

    lower = np.clip(tpr_mean - tpr_std, 0.0, 1.0)
    upper = np.clip(tpr_mean + tpr_std, 0.0, 1.0)

    plt.figure(figsize=(7.2, 5.8))
    plt.plot(fpr_grid, tpr_mean, linewidth=2.2, label="Mean ROC")
    plt.fill_between(fpr_grid, lower, upper, alpha=0.25, label="Std band")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, color="gray")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"Final ROC curve, {model_label}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()


def save_theorem_scatter(
    df_points: pd.DataFrame,
    model_key: str,
    model_label: str,
    outpath: Path,
) -> None:
    """
    Save IG_step versus IG_min scatter for flipped validation points only.
    """
    df_flip = df_points[
        (df_points["model"] == model_key) &
        (df_points["flip"] == 1)
    ].copy()

    if df_flip.empty:
        print(f"No flipped points found for {model_key}; skipping {outpath.name}")
        return

    x = df_flip["ig_min"].to_numpy()
    y = df_flip["ig_step"].to_numpy()

    maxv = float(np.nanmax([np.nanmax(x), np.nanmax(y)]))
    if not np.isfinite(maxv) or maxv <= 0:
        maxv = 1.0

    plt.figure(figsize=(7.2, 5.8))
    plt.scatter(x, y, s=8, alpha=0.35)
    plt.plot([0, maxv], [0, maxv], linestyle="--", linewidth=1.5, color="black")
    plt.xlabel(r"$IG_{\min}=2|p_{\mathrm{prev}}-\tau|^2$")
    plt.ylabel(
        r"$IG_{\mathrm{step}}="
        r"D_{\mathrm{KL}}(\mathrm{Bern}(p_{\mathrm{curr}})"
        r"\,\|\,\mathrm{Bern}(p_{\mathrm{prev}}))$"
    )
    plt.title(f"Stability-certificate diagnostic, {model_label}")
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()


def aggregate_summary(df_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate repetition-level outputs by model and training size.
    """
    df_ok = df_summary[df_summary["skipped_single_class_prefix"] == 0].copy()

    agg = (
        df_ok
        .groupby(["model", "train_size"], as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            precision_mean=("precision", "mean"),
            precision_std=("precision", "std"),
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
            specificity_mean=("specificity", "mean"),
            specificity_std=("specificity", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            auprc_mean=("auprc", "mean"),
            auprc_std=("auprc", "std"),
            brier_mean=("brier", "mean"),
            brier_std=("brier", "std"),
            flip_rate_mean=("flip_rate_step", "mean"),
            flip_rate_std=("flip_rate_step", "std"),
            ig_step_mean=("ig_step_mean", "mean"),
            ig_step_std=("ig_step_mean", "std"),
            ratio_median_mean=("ratio_flips_median", "mean"),
            ratio_median_std=("ratio_flips_median", "std"),
            violations_sum=("violations_step", "sum"),
            flips_sum=("flips_step", "sum"),
        )
    )

    return agg


def make_manuscript_figures(
    agg: pd.DataFrame,
    df_points: pd.DataFrame,
    final_roc_curves: dict[str, list[tuple[np.ndarray, np.ndarray]]],
) -> None:
    """
    Generate all figure files 
    """
    model_info = {
        "ann": {
            "label": "ANN",
            "suffix": "ann",
        },
        "xgb": {
            "label": "XGBoost",
            "suffix": "xgb",
        },
    }

    metric_plot_specs = {
        "flip_rate": {
            "mean_col": "flip_rate_mean",
            "std_col": "flip_rate_std",
            "ylabel": "Flip rate",
            "title": "Step-to-step flip rate",
            "filename": "fig_flip_rate_{suffix}.png",
        },
        "ig": {
            "mean_col": "ig_step_mean",
            "std_col": "ig_step_std",
            "ylabel": r"Mean $IG_{\mathrm{step}}$",
            "title": "Mean step-to-step KL displacement",
            "filename": "fig_ig_{suffix}.png",
        },
        "auroc": {
            "mean_col": "auc_mean",
            "std_col": "auc_std",
            "ylabel": "AUROC",
            "title": "AUROC",
            "filename": "fig_auroc_{suffix}.png",
        },
        "auprc": {
            "mean_col": "auprc_mean",
            "std_col": "auprc_std",
            "ylabel": "AUPRC",
            "title": "AUPRC",
            "filename": "fig_auprc_{suffix}.png",
        },
        "precision": {
            "mean_col": "precision_mean",
            "std_col": "precision_std",
            "ylabel": "Precision",
            "title": "Precision",
            "filename": "fig_precision_{suffix}.png",
        },
        "recall": {
            "mean_col": "recall_mean",
            "std_col": "recall_std",
            "ylabel": "Recall",
            "title": "Recall",
            "filename": "fig_recall_{suffix}.png",
        },
        "specificity": {
            "mean_col": "specificity_mean",
            "std_col": "specificity_std",
            "ylabel": "Specificity",
            "title": "Specificity",
            "filename": "fig_specificity_{suffix}.png",
        },
        "f1": {
            "mean_col": "f1_mean",
            "std_col": "f1_std",
            "ylabel": "F1 score",
            "title": "F1 score",
            "filename": "fig_f1_{suffix}.png",
        },
        "brier": {
            "mean_col": "brier_mean",
            "std_col": "brier_std",
            "ylabel": "Brier score",
            "title": "Brier score",
            "filename": "fig_brier_{suffix}.png",
        },
    }

    for model_key, info in model_info.items():
        model_label = info["label"]
        suffix = info["suffix"]

        df_m = agg[agg["model"] == model_key].copy()
        if df_m.empty:
            print(f"No aggregate rows for {model_key}; skipping.")
            continue

        x = df_m["train_size"].to_numpy(dtype=float)

        for spec in metric_plot_specs.values():
            y_mean = df_m[spec["mean_col"]].to_numpy(dtype=float)
            y_std = df_m[spec["std_col"]].fillna(0.0).to_numpy(dtype=float)

            outpath = FIGURES_DIR / spec["filename"].format(suffix=suffix)
            plot_mean_std(
                x=x,
                y_mean=y_mean,
                y_std=y_std,
                ylabel=spec["ylabel"],
                title=f"{spec['title']}, {model_label}",
                outpath=outpath,
            )

        save_theorem_scatter(
            df_points=df_points,
            model_key=model_key,
            model_label=model_label,
            outpath=FIGURES_DIR / f"fig_theorem_{suffix}.png",
        )

        if model_key in final_roc_curves and len(final_roc_curves[model_key]) > 0:
            save_final_roc_figure(
                curves=final_roc_curves[model_key],
                model_label=model_label,
                outpath=FIGURES_DIR / f"fig_roc_{suffix}.png",
            )


# =============================================================================
# LOAD DATA
# =============================================================================

df = pd.read_csv(DATA_FILE).dropna().reset_index(drop=True)

if TARGET_COL is None:
    target_series = df.iloc[:, -1]
    feature_df = df.iloc[:, :-1].copy()
    target_name = df.columns[-1]
else:
    target_series = df[TARGET_COL]
    feature_df = df.drop(columns=[TARGET_COL]).copy()
    target_name = TARGET_COL

# PDAC.csv file has already been preprocessed numerically.
# If another file is to be used, encode nonnumeric columns before running this script.
try:
    X_full_raw = feature_df.astype(float).copy()
except ValueError as exc:
    raise ValueError(
        "All feature columns must be numeric in the preprocessed PDAC.csv file. "
        "Please encode categorical variables before running this script."
    ) from exc

y_full = coerce_binary_target(target_series)
feature_names = list(X_full_raw.columns)

rng_global = np.random.default_rng(RANDOM_SEED_GLOBAL)
perm = rng_global.permutation(len(df))
X_full_raw = X_full_raw.iloc[perm].reset_index(drop=True)
y_full = y_full[perm]

print("=" * 80)
print("PDAC decision-stability experiment")
print("=" * 80)
print(f"Input file: {DATA_FILE}")
print(f"Loaded data: n={len(df):,}, d={X_full_raw.shape[1]:,}, target='{target_name}'")
print(f"Class balance: mean(y)={y_full.mean():.3f} (fraction positive)")
print(f"Outputs: {BASE_DIR.resolve()}")
print("=" * 80)


# =============================================================================
# MAIN MONTE CARLO EXPERIMENT
# =============================================================================

summary_rows = []
theorem_points_rows = []

final_roc_curves = {
    "ann": [],
    "xgb": [],
}

for rep in range(N_REPETITIONS):
    print(f"===== Monte Carlo repetition {rep + 1}/{N_REPETITIONS} =====")

    idx_all = np.arange(len(y_full))
    idx_pool, idx_val = train_test_split(
        idx_all,
        test_size=VAL_SIZE,
        stratify=y_full,
        random_state=rep,
    )

    X_pool_raw = X_full_raw.iloc[idx_pool].reset_index(drop=True)
    y_pool = y_full[idx_pool]
    X_val_raw = X_full_raw.iloc[idx_val].reset_index(drop=True)
    y_val = y_full[idx_val]

    # Fit scaler on the training pool only. This avoids validation leakage.
    scaler = MinMaxScaler()
    X_pool = scaler.fit_transform(X_pool_raw.values)
    X_val = scaler.transform(X_val_raw.values)

    rng_rep = np.random.default_rng(rep)
    order = stratified_progressive_order(y_pool, rng_rep)
    X_pool = X_pool[order]
    y_pool = y_pool[order]

    if INITIAL_TRAIN_SIZE >= len(y_pool):
        raise ValueError(
            f"INITIAL_TRAIN_SIZE={INITIAL_TRAIN_SIZE} must be smaller than "
            f"training-pool size={len(y_pool)}."
        )

    train_sizes = np.arange(INITIAL_TRAIN_SIZE, len(y_pool) + 1, STEP_SIZE)

    models = build_models(random_state=rep)

    for model_name, model in models.items():
        print(f"  Model: {model_name}")

        prev_proba = None
        prev_n = None

        last_y_proba = None

        for n in train_sizes:
            X_train = X_pool[:n]
            y_train = y_pool[:n]

            if len(np.unique(y_train)) < 2:
                summary_rows.append({
                    "rep": rep,
                    "model": model_name,
                    "train_size": int(n),
                    "skipped_single_class_prefix": 1,
                })
                continue

            model.fit(X_train, y_train)

            y_proba = model.predict_proba(X_val)[:, 1].astype(float)
            y_proba = np.clip(y_proba, EPS, 1.0 - EPS)
            y_pred = (y_proba >= TAU).astype(int)

            metrics = compute_metrics_from_predictions(
                y_true=y_val,
                y_pred=y_pred,
                y_proba=y_proba,
            )

            if prev_proba is None:
                flips_step = np.nan
                violations_step = np.nan
                ig_step_mean = np.nan
                ig_step_median = np.nan
                ig_step_flips_mean = np.nan
                ig_min_flips_mean = np.nan
                ratio_flips_min = np.nan
                ratio_flips_median = np.nan
                ratio_flips_p10 = np.nan
                ratio_flips_p90 = np.nan
            else:
                p_prev = np.clip(prev_proba, EPS, 1.0 - EPS)
                p_curr = np.clip(y_proba, EPS, 1.0 - EPS)

                ig_step = bern_kl(p_curr, p_prev, eps=EPS)
                ig_min = 2.0 * (np.abs(p_prev - TAU) ** 2)

                u_prev = (p_prev >= TAU).astype(int)
                u_curr = (p_curr >= TAU).astype(int)
                flip = u_prev != u_curr

                violations_step = int(np.sum(flip & (ig_step + VIOL_TOL < ig_min)))
                flips_step = int(np.sum(flip))

                ig_step_mean = float(np.mean(ig_step))
                ig_step_median = float(np.median(ig_step))

                if flips_step > 0:
                    ratio = ig_step[flip] / np.maximum(ig_min[flip], EPS)

                    ig_step_flips_mean = float(np.mean(ig_step[flip]))
                    ig_min_flips_mean = float(np.mean(ig_min[flip]))
                    ratio_flips_min = float(np.min(ratio))
                    ratio_flips_median = float(np.median(ratio))
                    ratio_flips_p10 = float(np.percentile(ratio, 10))
                    ratio_flips_p90 = float(np.percentile(ratio, 90))
                else:
                    ig_step_flips_mean = np.nan
                    ig_min_flips_mean = np.nan
                    ratio_flips_min = np.nan
                    ratio_flips_median = np.nan
                    ratio_flips_p10 = np.nan
                    ratio_flips_p90 = np.nan

                for i in range(len(y_val)):
                    theorem_points_rows.append({
                        "rep": rep,
                        "model": model_name,
                        "train_size_prev": int(prev_n),
                        "train_size_curr": int(n),
                        "val_row": int(i),
                        "y_true": int(y_val[i]),
                        "p_prev": float(p_prev[i]),
                        "p_curr": float(p_curr[i]),
                        "margin_prev": float(abs(p_prev[i] - TAU)),
                        "flip": int(flip[i]),
                        "ig_step": float(ig_step[i]),
                        "ig_min": float(ig_min[i]),
                        "ratio": float(ig_step[i] / max(ig_min[i], EPS)),
                        "violates": int((ig_step[i] + VIOL_TOL < ig_min[i]) and flip[i]),
                    })

            summary_rows.append({
                "rep": rep,
                "model": model_name,
                "train_size": int(n),
                "skipped_single_class_prefix": 0,

                **metrics,

                "train_size_prev": int(prev_n) if prev_n is not None else np.nan,
                "flips_step": flips_step,
                "flip_rate_step": float(flips_step / len(y_val))
                if isinstance(flips_step, int) else np.nan,
                "violations_step": violations_step,
                "ig_step_mean": ig_step_mean,
                "ig_step_median": ig_step_median,
                "ig_step_flips_mean": ig_step_flips_mean,
                "ig_min_flips_mean": ig_min_flips_mean,
                "ratio_flips_min": ratio_flips_min,
                "ratio_flips_median": ratio_flips_median,
                "ratio_flips_p10": ratio_flips_p10,
                "ratio_flips_p90": ratio_flips_p90,
            })

            prev_proba = y_proba.copy()
            prev_n = int(n)
            last_y_proba = y_proba.copy()

        if last_y_proba is not None and len(np.unique(y_val)) == 2:
            fpr, tpr, _ = roc_curve(y_val, last_y_proba)
            final_roc_curves[model_name].append((fpr, tpr))


# =============================================================================
# SAVE CSV OUTPUTS
# =============================================================================

df_summary = pd.DataFrame(summary_rows)
df_theorem = pd.DataFrame(theorem_points_rows)

summary_path = BASE_DIR / "progressive_mc_results_summary.csv"
theorem_path = BASE_DIR / "theorem_points_long.csv"

df_summary.to_csv(summary_path, index=False)
df_theorem.to_csv(theorem_path, index=False)

agg = aggregate_summary(df_summary)
agg_path = TABLES_DIR / "aggregated_stability_tables.csv"
agg.to_csv(agg_path, index=False)

# Overall stability summary for manuscript Table 3.
df_flip = df_theorem[df_theorem["flip"] == 1].copy()
overall_rows = []

for model_name in ["ann", "xgb"]:
    df_model_all = df_theorem[df_theorem["model"] == model_name]
    df_model_flip = df_flip[df_flip["model"] == model_name]

    total_comparisons = len(df_model_all)
    total_flips = len(df_model_flip)
    flip_incidence = total_flips / total_comparisons if total_comparisons > 0 else np.nan
    violations = int(df_model_flip["violates"].sum()) if total_flips > 0 else 0

    overall_rows.append({
        "model": model_name,
        "total_comparisons": total_comparisons,
        "total_flips": total_flips,
        "flip_incidence": flip_incidence,
        "violations": violations,
        "rho_min": float(df_model_flip["ratio"].min()) if total_flips > 0 else np.nan,
        "rho_median": float(df_model_flip["ratio"].median()) if total_flips > 0 else np.nan,
        "rho_p10": float(df_model_flip["ratio"].quantile(0.10)) if total_flips > 0 else np.nan,
        "rho_p90": float(df_model_flip["ratio"].quantile(0.90)) if total_flips > 0 else np.nan,
    })

overall = pd.DataFrame(overall_rows)
overall_path = TABLES_DIR / "overall_stability_summary.csv"
overall.to_csv(overall_path, index=False)


# =============================================================================
# GENERATE MANUSCRIPT FIGURES
# =============================================================================

make_manuscript_figures(
    agg=agg,
    df_points=df_theorem,
    final_roc_curves=final_roc_curves,
)


# =============================================================================
# COMPLETION MESSAGE
# =============================================================================

print("=" * 80)
print("All runs complete.")
print("=" * 80)
print(f"Summary CSV: {summary_path}")
print(f"Patient-level stability CSV: {theorem_path}")
print(f"Aggregated table CSV: {agg_path}")
print(f"Overall stability CSV: {overall_path}")
print(f"Manuscript figures directory: {FIGURES_DIR}")
print("=" * 80)
print("Generated manuscript figure files:")
for path in sorted(FIGURES_DIR.glob("fig_*.png")):
    print(f"  {path.name}")
print("=" * 80)
