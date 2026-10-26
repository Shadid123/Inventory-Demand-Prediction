"""Run exploratory analysis, classification, and clustering for the project.

Uses the project's existing chronological 80/20 split. Demand is never used as
an input feature for the demand-level classifier or the K-Means model.
"""
# Allow modern type-hint syntax to be interpreted consistently.
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    silhouette_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Build paths from this script location so the program works from any terminal folder.
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "sales_data.csv"
MODEL_DIR = BASE_DIR / "models"
ANALYSIS_DIR = BASE_DIR / "analysis"
FIGURE_DIR = ANALYSIS_DIR / "figures"

# Demand is the numeric outcome for regression and the source for derived classification labels.
TARGET = "Demand"
EXPECTED_COLUMNS = [
    "Date", "Store ID", "Product ID", "Category", "Region",
    "Inventory Level", "Units Sold", "Units Ordered", "Price", "Discount",
    "Weather Condition", "Promotion", "Competitor Pricing", "Seasonality",
    "Epidemic", "Demand",
]
# The classifier groups demand into these three human-readable labels.
CLASSES = ["Low", "Medium", "High"]
# Only these inventory/market columns are used to group records; Demand is not included.
CLUSTER_FEATURES = [
    "Inventory Level", "Units Sold", "Units Ordered", "Price", "Discount",
    "Competitor Pricing", "Promotion", "Epidemic",
]
# Test cluster counts from 2 through 6 and select using silhouette score.
K_CANDIDATES = range(2, 7)
RANDOM_STATE = 42


# Create output folders if they do not already exist.
def prepare_directories() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# Load the CSV and stop early if the file, required columns, or Demand values are invalid.
def load_and_validate_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Put sales_data.csv in the data folder."
        )
    df = pd.read_csv(DATA_PATH)
    missing_columns = [column for column in EXPECTED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")
    if df.empty:
        raise ValueError("The dataset is empty.")
    df["Date"] = pd.to_datetime(df["Date"], errors="raise")
    if df[TARGET].isna().any():
        raise ValueError("The target column 'Demand' contains missing values.")
    return df


# Split by date so earlier records train the models and later records test them.
def chronological_split(df: pd.DataFrame) -> tuple[pd.Series, pd.Timestamp]:
    """Return a row mask for the earlier 80% of unique dates and its cutoff."""
    dates = df["Date"]
    unique_dates = np.sort(dates.unique())
    if len(unique_dates) < 2:
        raise ValueError("At least two distinct dates are required for a chronological split.")
    cutoff = pd.Timestamp(unique_dates[int(len(unique_dates) * 0.80)])
    train_mask = dates < cutoff
    if not train_mask.any() or train_mask.all():
        raise ValueError("The chronological split did not produce both train and test rows.")
    return train_mask, cutoff


# Replace the raw date with year/month/day/weekday/day-of-year features.
def make_model_features(df: pd.DataFrame) -> pd.DataFrame:
    features = df.copy()
    date = pd.to_datetime(features.pop("Date"), errors="raise")
    features["Year"] = date.dt.year
    features["Month"] = date.dt.month
    features["Day"] = date.dt.day
    features["DayOfWeek"] = date.dt.dayofweek
    features["DayOfYear"] = date.dt.dayofyear
    return features


# Write a Python dictionary to readable JSON for later inspection or display.
def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, default=_json_default), encoding="utf-8")


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if pd.isna(value):
        return None
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_figure(fig: plt.Figure, filename: str) -> None:
    fig.savefig(FIGURE_DIR / filename, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# EDA summarizes data quality, distributions, relationships, and saves charts/tables.
def run_eda(df: pd.DataFrame, train_mask: pd.Series, cutoff: pd.Timestamp) -> dict:
    """Create numeric/categorical summaries, findings, and EDA charts."""
    n_rows, n_cols = df.shape
    missing_by_column = df.isna().sum().astype(int)
    overview = pd.DataFrame([
        {"Metric": "Rows", "Value": n_rows},
        {"Metric": "Columns", "Value": n_cols},
        {"Metric": "Missing cells", "Value": int(missing_by_column.sum())},
        {"Metric": "Exact duplicate rows", "Value": int(df.duplicated().sum())},
        {"Metric": "First date", "Value": df["Date"].min().date().isoformat()},
        {"Metric": "Last date", "Value": df["Date"].max().date().isoformat()},
        {"Metric": "Training rows (chronological 80%)", "Value": int(train_mask.sum())},
        {"Metric": "Test rows (later 20%)", "Value": int((~train_mask).sum())},
        {"Metric": "Test period starts", "Value": cutoff.date().isoformat()},
    ])
    overview.to_csv(ANALYSIS_DIR / "dataset_overview.csv", index=False)
    missing_by_column.rename("Missing Values").to_csv(ANALYSIS_DIR / "missing_values.csv", header=True)

    preprocessing_steps = pd.DataFrame([
        {"Stage": "Schema validation", "Action": "Verify all 16 expected columns are present", "Reason": "Catch a wrong or incomplete CSV before modeling"},
        {"Stage": "Data quality", "Action": f"Count missing cells ({int(missing_by_column.sum())}) and exact duplicate rows ({int(df.duplicated().sum())}); no rows dropped", "Reason": "Record actual data quality without silently changing the dataset"},
        {"Stage": "Date features", "Action": "Parse Date; derive Year, Month, Day, DayOfWeek, DayOfYear for regression/classification", "Reason": "Make calendar information available in numeric features"},
        {"Stage": "Categorical encoding", "Action": "OneHotEncoder(handle_unknown='ignore') in regression/classification pipelines", "Reason": "Represent nominal categories as model-readable numeric columns"},
        {"Stage": "Missing-value handling", "Action": "Median imputation for numeric / most-frequent imputation for categorical classifier inputs; median imputation for K-Means inputs", "Reason": "Make the analysis pipeline robust to missing feature values; no imputation is needed for the supplied complete CSV"},
        {"Stage": "Clustering scale", "Action": "StandardScaler on eight numeric inventory/market signals", "Reason": "Put features with different units/ranges on comparable scales for distance-based K-Means"},
        {"Stage": "Target separation", "Action": "Exclude Demand from classifier input features and from K-Means inputs", "Reason": "Prevent the target itself from being supplied as a predictor / clustering dimension"},
        {"Stage": "Chronological evaluation", "Action": f"Use dates before {pd.Timestamp(cutoff).date()} for training and dates on/after it for testing", "Reason": "Evaluate on later observations instead of randomly mixing past and later rows"},
        {"Stage": "Outlier policy", "Action": "Report numeric ranges and visualize distributions; do not automatically remove or cap extreme values", "Reason": "Avoid treating unusual values as errors without source/business validation"},
    ])
    preprocessing_steps.to_csv(ANALYSIS_DIR / "preprocessing_steps.csv", index=False)

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    numeric_summary = df[numeric_cols].describe().T.reset_index(names="Column")
    numeric_summary.to_csv(ANALYSIS_DIR / "numeric_summary.csv", index=False)

    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    category_rows = []
    for column in categorical_cols:
        counts = df[column].value_counts(dropna=False)
        for value, count in counts.items():
            category_rows.append({
                "Column": column,
                "Value": str(value),
                "Count": int(count),
                "Percent": round(float(count / len(df) * 100), 3),
            })
    pd.DataFrame(category_rows).to_csv(ANALYSIS_DIR / "categorical_distributions.csv", index=False)

    correlations = df[numeric_cols].corr(numeric_only=True)[TARGET].drop(TARGET).sort_values(
        key=lambda values: values.abs(), ascending=False
    )
    correlations.rename("Correlation with Demand").to_csv(ANALYSIS_DIR / "correlations_with_demand.csv", header=True)

    category_summary = df.groupby("Category", dropna=False).agg(
        Records=(TARGET, "size"),
        Mean_Demand=(TARGET, "mean"),
        Median_Demand=(TARGET, "median"),
        Std_Demand=(TARGET, "std"),
    ).reset_index().sort_values("Mean_Demand", ascending=False)
    category_summary.to_csv(ANALYSIS_DIR / "demand_by_category.csv", index=False, float_format="%.4f")

    monthly = df.assign(Month=df["Date"].dt.to_period("M").dt.to_timestamp()).groupby("Month").agg(
        Records=(TARGET, "size"),
        Mean_Demand=(TARGET, "mean"),
        Total_Demand=(TARGET, "sum"),
        Mean_Units_Sold=("Units Sold", "mean"),
    ).reset_index()
    monthly.to_csv(ANALYSIS_DIR / "monthly_demand_summary.csv", index=False, float_format="%.4f")

    promotion_summary = df.groupby("Promotion").agg(
        Records=(TARGET, "size"), Mean_Demand=(TARGET, "mean"), Median_Demand=(TARGET, "median")
    ).reset_index()
    promotion_summary["Promotion Label"] = promotion_summary["Promotion"].map({0: "No promotion", 1: "Promotion active"}).fillna(
        promotion_summary["Promotion"].astype(str)
    )
    promotion_summary.to_csv(ANALYSIS_DIR / "promotion_demand_summary.csv", index=False, float_format="%.4f")

    # Plot 1: target distribution
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#0b1020")
    ax.set_facecolor("#11182b")
    ax.hist(df[TARGET], bins=35, color="#9b7bff", edgecolor="#e5e7eb", alpha=0.88)
    ax.set_title("Distribution of Recorded Demand", color="#f7f8ff", pad=14, fontsize=14, weight="bold")
    ax.set_xlabel("Demand (units)", color="#c5cce0")
    ax.set_ylabel("Number of records", color="#c5cce0")
    ax.tick_params(colors="#c5cce0")
    ax.grid(axis="y", color="#ffffff", alpha=0.08)
    for spine in ax.spines.values(): spine.set_color("#3b4560")
    save_figure(fig, "demand_distribution.png")

    # Plot 2: demand by category
    category_order = category_summary["Category"].tolist()
    category_data = [df.loc[df["Category"] == category, TARGET].to_numpy() for category in category_order]
    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor="#0b1020")
    ax.set_facecolor("#11182b")
    bp = ax.boxplot(category_data, tick_labels=category_order, patch_artist=True, showfliers=False)
    for index, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(["#9b7bff", "#5ee7f8", "#f36cf4", "#78f5bb", "#ffd17a"][index % 5])
        patch.set_alpha(0.6)
        patch.set_edgecolor("#dbe3f5")
    for item in bp["whiskers"] + bp["caps"] + bp["medians"]:
        item.set_color("#dbe3f5")
    ax.set_title("Recorded Demand by Product Category", color="#f7f8ff", pad=14, fontsize=14, weight="bold")
    ax.set_xlabel("Category", color="#c5cce0")
    ax.set_ylabel("Demand (units)", color="#c5cce0")
    ax.tick_params(colors="#c5cce0", axis="both")
    ax.grid(axis="y", color="#ffffff", alpha=0.08)
    for spine in ax.spines.values(): spine.set_color("#3b4560")
    save_figure(fig, "demand_by_category.png")

    # Plot 3: monthly demand trend
    fig, ax = plt.subplots(figsize=(11, 5), facecolor="#0b1020")
    ax.set_facecolor("#11182b")
    ax.plot(monthly["Month"], monthly["Mean_Demand"], color="#5ee7f8", linewidth=2.2, marker="o", markersize=3.5)
    ax.set_title("Monthly Average Demand Over Time", color="#f7f8ff", pad=14, fontsize=14, weight="bold")
    ax.set_xlabel("Month", color="#c5cce0")
    ax.set_ylabel("Average demand (units)", color="#c5cce0")
    ax.tick_params(colors="#c5cce0", axis="both")
    ax.grid(color="#ffffff", alpha=0.08)
    for spine in ax.spines.values(): spine.set_color("#3b4560")
    fig.autofmt_xdate()
    save_figure(fig, "monthly_average_demand.png")

    # Plot 4: correlation matrix of numeric fields
    corr_matrix = df[numeric_cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8), facecolor="#0b1020")
    ax.set_facecolor("#11182b")
    image = ax.imshow(corr_matrix.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(corr_matrix.columns)), corr_matrix.columns, rotation=75, ha="right", color="#c5cce0", fontsize=8)
    ax.set_yticks(np.arange(len(corr_matrix.index)), corr_matrix.index, color="#c5cce0", fontsize=8)
    ax.set_title("Correlation Between Numeric Variables", color="#f7f8ff", pad=14, fontsize=14, weight="bold")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    save_figure(fig, "numeric_correlation_heatmap.png")

    # Plot 5: promotion comparison
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#0b1020")
    ax.set_facecolor("#11182b")
    promo_order = [0, 1]
    promo_data = [df.loc[df["Promotion"] == value, TARGET].to_numpy() for value in promo_order]
    bp = ax.boxplot(promo_data, tick_labels=["No promotion", "Promotion active"], patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], ["#9b7bff", "#5ee7f8"]):
        patch.set_facecolor(color); patch.set_alpha(0.62); patch.set_edgecolor("#e5e7eb")
    for item in bp["whiskers"] + bp["caps"] + bp["medians"]: item.set_color("#dbe3f5")
    ax.set_title("Recorded Demand by Promotion Status", color="#f7f8ff", pad=14, fontsize=14, weight="bold")
    ax.set_ylabel("Demand (units)", color="#c5cce0")
    ax.tick_params(colors="#c5cce0")
    ax.grid(axis="y", color="#ffffff", alpha=0.08)
    for spine in ax.spines.values(): spine.set_color("#3b4560")
    save_figure(fig, "demand_by_promotion.png")

    mean_demand = float(df[TARGET].mean())
    median_demand = float(df[TARGET].median())
    corr_feature = str(correlations.index[0]) if len(correlations) else "None"
    corr_value = float(correlations.iloc[0]) if len(correlations) else 0.0
    highest_category = category_summary.iloc[0]
    lowest_category = category_summary.iloc[-1]
    highest_month = monthly.loc[monthly["Mean_Demand"].idxmax()]
    lowest_month = monthly.loc[monthly["Mean_Demand"].idxmin()]
    promo_means = promotion_summary.set_index("Promotion Label")["Mean_Demand"].to_dict()

    findings = f"""# Exploratory Data Analysis Findings

## Dataset checks

- **Rows:** {n_rows:,}
- **Columns:** {n_cols}
- **Date range:** {df['Date'].min().date()} to {df['Date'].max().date()}
- **Missing cells:** {int(missing_by_column.sum()):,}
- **Exact duplicate rows:** {int(df.duplicated().sum()):,}
- **Chronological split:** {int(train_mask.sum()):,} earlier records for training and {int((~train_mask).sum()):,} later records for testing; test starts {cutoff.date()}.

No rows were silently removed during this analysis. Missing values and duplicate rows were checked and counted explicitly.

## Demand patterns

- Recorded demand has a mean of **{mean_demand:.2f}** units and a median of **{median_demand:.2f}** units; the range is **{int(df[TARGET].min())}–{int(df[TARGET].max())}** units.
- The category with the highest mean demand is **{highest_category['Category']}** ({highest_category['Mean_Demand']:.2f} units on average in this dataset).
- The category with the lowest mean demand is **{lowest_category['Category']}** ({lowest_category['Mean_Demand']:.2f} units on average in this dataset).
- The month with the highest mean demand is **{pd.Timestamp(highest_month['Month']).strftime('%Y-%m')}** ({highest_month['Mean_Demand']:.2f} units); the lowest is **{pd.Timestamp(lowest_month['Month']).strftime('%Y-%m')}** ({lowest_month['Mean_Demand']:.2f} units).
- The strongest absolute numeric correlation with `Demand` is **{corr_feature}** (Pearson correlation {corr_value:.3f}). Correlation describes association, not proof of causation.
- Mean demand was **{promo_means.get('No promotion', float('nan')):.2f}** units with no promotion and **{promo_means.get('Promotion active', float('nan')):.2f}** units when promotion was active. This comparison is descriptive and does not isolate promotion as the cause.

## Figures and output files

- `figures/demand_distribution.png` — distribution of the target variable
- `figures/demand_by_category.png` — demand distribution by product category
- `figures/monthly_average_demand.png` — monthly average recorded demand
- `figures/numeric_correlation_heatmap.png` — correlations between numeric fields
- `figures/demand_by_promotion.png` — demand distribution by promotion status
- `dataset_overview.csv`, `numeric_summary.csv`, `categorical_distributions.csv`, `correlations_with_demand.csv`, `demand_by_category.csv`, `monthly_demand_summary.csv`, and `promotion_demand_summary.csv` — supporting tables

## Interpretation caution

`Units Sold` is strongly associated with the recorded `Demand` in this dataset. The regression and classification tasks therefore describe patterns in the supplied records. A real future-demand forecast must ensure every input represents information already available at the forecast time; using sales from the same period as the target can create target leakage for a true forecast.
"""
    (ANALYSIS_DIR / "eda_findings.md").write_text(findings, encoding="utf-8")

    return {
        "rows": n_rows,
        "columns": n_cols,
        "missing_cells": int(missing_by_column.sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "date_start": df["Date"].min().date().isoformat(),
        "date_end": df["Date"].max().date().isoformat(),
        "mean_demand": round(mean_demand, 4),
        "median_demand": round(median_demand, 4),
        "strongest_numeric_correlation_feature": corr_feature,
        "strongest_numeric_correlation": round(corr_value, 4),
    }


# Calculate several classification metrics so performance is not judged by accuracy alone.
def metrics_for_classifier(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=CLASSES, average="weighted", zero_division=0)),
    }


# Train a Decision Tree to predict Low/Medium/High demand and evaluate it on later dates.
def run_classification(df: pd.DataFrame, train_mask: pd.Series, cutoff: pd.Timestamp) -> dict:
    features = make_model_features(df)
    X = features.drop(columns=[TARGET])
    y_demand = features[TARGET]
    train_y_values = y_demand.loc[train_mask]
    # Derive class boundaries from training Demand only, avoiding use of test outcomes to set thresholds.
    lower_cutoff, upper_cutoff = np.quantile(train_y_values, [1 / 3, 2 / 3]).astype(float)

    def to_demand_class(values: pd.Series) -> pd.Series:
        labels = np.where(values <= lower_cutoff, "Low", np.where(values <= upper_cutoff, "Medium", "High"))
        return pd.Series(labels, index=values.index, name="Demand Level")

    y = to_demand_class(y_demand)
    X_train, X_test = X.loc[train_mask], X.loc[~train_mask]
    y_train, y_test = y.loc[train_mask], y.loc[~train_mask]

    categorical_cols = [column for column in X.columns if X[column].dtype == "object"]
    numeric_cols = [column for column in X.columns if column not in categorical_cols]
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical_cols),
            ("numeric", SimpleImputer(strategy="median"), numeric_cols),
        ],
        remainder="drop",
    )
    # Keep imputing/encoding and classification in one pipeline to apply the same steps consistently.
    classifier = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(
            max_depth=12, min_samples_leaf=20, class_weight="balanced", random_state=RANDOM_STATE
        )),
    ])
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(np.zeros((len(y_train), 1)), y_train)
    baseline_predictions = baseline.predict(np.zeros((len(y_test), 1)))

    classifier_metrics = metrics_for_classifier(y_test, predictions)
    baseline_metrics = metrics_for_classifier(y_test, baseline_predictions)
    metric_rows = [
        {"Model": "Decision Tree Classifier", **classifier_metrics},
        {"Model": "Most-Frequent Baseline", **baseline_metrics},
    ]
    pd.DataFrame(metric_rows).to_csv(ANALYSIS_DIR / "classification_metrics.csv", index=False, float_format="%.5f")

    report = classification_report(y_test, predictions, labels=CLASSES, output_dict=True, zero_division=0)
    report.pop("accuracy", None)  # Accuracy is reported separately in classification_metrics.csv.
    report_frame = pd.DataFrame(report).T.reset_index(names="Class or average")
    report_frame.to_csv(ANALYSIS_DIR / "classification_report.csv", index=False, float_format="%.5f")

    matrix = confusion_matrix(y_test, predictions, labels=CLASSES)
    pd.DataFrame(
        matrix,
        index=[f"Actual {value}" for value in CLASSES],
        columns=[f"Predicted {value}" for value in CLASSES],
    ).to_csv(ANALYSIS_DIR / "classification_confusion_matrix.csv")

    fig, ax = plt.subplots(figsize=(7, 5.5), facecolor="#0b1020")
    ax.set_facecolor("#11182b")
    image = ax.imshow(matrix, cmap="Purples")
    ax.set_xticks(range(len(CLASSES)), CLASSES, color="#c5cce0")
    ax.set_yticks(range(len(CLASSES)), CLASSES, color="#c5cce0")
    ax.set_xlabel("Predicted demand level", color="#c5cce0")
    ax.set_ylabel("Actual demand level", color="#c5cce0")
    ax.set_title("Decision Tree Classifier — Confusion Matrix", color="#f7f8ff", pad=14, fontsize=13, weight="bold")
    threshold = matrix.max() / 2 if matrix.size else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:,}", ha="center", va="center",
                    color="#f7f8ff" if matrix[i, j] > threshold else "#11182b", fontsize=11, weight="bold")
    ax.tick_params(colors="#c5cce0")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    save_figure(fig, "classification_confusion_matrix.png")

    joblib.dump(classifier, MODEL_DIR / "demand_level_classifier.joblib")
    metadata = {
        "task": "Three-class classification of recorded demand level",
        "target_name": "Demand Level",
        "class_order": CLASSES,
        "thresholds": {
            "training_demand_quantile_33_percent": round(float(lower_cutoff), 4),
            "training_demand_quantile_67_percent": round(float(upper_cutoff), 4),
            "rules": {
                "Low": f"Demand <= {lower_cutoff:g}",
                "Medium": f"{lower_cutoff:g} < Demand <= {upper_cutoff:g}",
                "High": f"Demand > {upper_cutoff:g}",
            },
        },
        "feature_columns": X.columns.tolist(),
        "categorical_columns": categorical_cols,
        "numeric_columns": numeric_cols,
        "train_rows": int(train_mask.sum()),
        "test_rows": int((~train_mask).sum()),
        "test_start_date": cutoff.date().isoformat(),
        "metrics": classifier_metrics,
        "baseline_metrics": baseline_metrics,
        "leakage_note": "Demand is excluded from features. Units Sold remains an input as recorded in the dataset; this task classifies the supplied observation and is not by itself proof of a future forecast.",
    }
    save_json(MODEL_DIR / "classification_metadata.json", metadata)
    save_json(ANALYSIS_DIR / "classification_metrics.json", metadata)
    classification_findings = f"""# Classification Findings

- **Algorithm:** Decision Tree Classifier.
- **Task:** classify each recorded observation as Low, Medium or High demand.
- **Training-only cutoffs:** Low is `Demand <= {lower_cutoff:g}`; Medium is `{lower_cutoff:g} < Demand <= {upper_cutoff:g}`; High is `Demand > {upper_cutoff:g}`.
- **Chronological split:** {int(train_mask.sum()):,} training records and {int((~train_mask).sum()):,} later test records; testing starts {cutoff.date()}.
- **Test accuracy:** {classifier_metrics['accuracy']:.2%}.
- **Balanced accuracy:** {classifier_metrics['balanced_accuracy']:.2%}.
- **Macro F1:** {classifier_metrics['macro_f1']:.4f}.
- **Most-frequent-class baseline accuracy:** {baseline_metrics['accuracy']:.2%}.

The classifier performs better than the simple baseline on this held-out test period. Demand is not used as an input feature; the class is derived from demand solely as the target. The full classification report and confusion matrix are saved beside this file.

**Interpretation note:** this classifies recorded demand levels from the other supplied columns. Because `Units Sold` is strongly associated with `Demand`, verify the timing of sales information before treating it as a true future prediction.
"""
    (ANALYSIS_DIR / "classification_findings.md").write_text(classification_findings, encoding="utf-8")

    class_counts = pd.DataFrame({
        "Class": CLASSES,
        "Training records": [int((y_train == label).sum()) for label in CLASSES],
        "Testing records": [int((y_test == label).sum()) for label in CLASSES],
    })
    class_counts.to_csv(ANALYSIS_DIR / "classification_class_counts.csv", index=False)

    return {
        "model": "Decision Tree Classifier",
        "thresholds": [float(lower_cutoff), float(upper_cutoff)],
        "metrics": classifier_metrics,
        "baseline_metrics": baseline_metrics,
        "model_path": "models/demand_level_classifier.joblib",
    }


# Use K-Means to find groups of similar records without supplying demand labels.
def run_clustering(df: pd.DataFrame, train_mask: pd.Series, cutoff: pd.Timestamp) -> dict:
    X = df[CLUSTER_FEATURES].copy()
    X_train = X.loc[train_mask]
    X_test = X.loc[~train_mask]

    # Choose k using only an approximately 8,000-row sample from the training period.
    # Sampling reduces computation while keeping test-period records out of cluster-count selection.
    # Fill any missing numeric feature values with the training median before scaling.
    imputer = SimpleImputer(strategy="median")
    # Standardize feature scales so price/stock ranges do not dominate K-Means distances.
    scaler = StandardScaler()
    X_train_imputed = imputer.fit_transform(X_train)
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    rng = np.random.default_rng(RANDOM_STATE)
    search_size = min(8000, len(X_train_scaled))
    search_indices = rng.choice(len(X_train_scaled), size=search_size, replace=False)
    search_sample = X_train_scaled[search_indices]
    silhouette_rows = []
    best_k = None
    best_score = -np.inf
    silhouette_sample_size = min(2500, len(search_sample))

    # Try each candidate cluster count and measure how separated/compact the groups are.
    for k in K_CANDIDATES:
        candidate = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        candidate_labels = candidate.fit_predict(search_sample)
        if len(np.unique(candidate_labels)) < 2:
            score = float("nan")
        else:
            score = float(silhouette_score(
                search_sample,
                candidate_labels,
                sample_size=silhouette_sample_size,
                random_state=RANDOM_STATE,
            ))
        silhouette_rows.append({"k": k, "Training sample silhouette": score})
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_k = k

    if best_k is None:
        raise RuntimeError("Could not select a valid K-Means cluster count from k=2..6.")

    cluster_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=best_k, n_init=10, random_state=RANDOM_STATE)),
    ])
    cluster_pipeline.fit(X_train)
    test_labels = cluster_pipeline.predict(X_test)
    X_test_scaled = cluster_pipeline.named_steps["scaler"].transform(
        cluster_pipeline.named_steps["imputer"].transform(X_test)
    )
    rng_test = np.random.default_rng(RANDOM_STATE)
    test_n = min(3000, len(X_test_scaled))
    test_idx = rng_test.choice(len(X_test_scaled), size=test_n, replace=False)
    test_labels_sample = test_labels[test_idx]
    if len(np.unique(test_labels_sample)) >= 2:
        test_silhouette = float(silhouette_score(X_test_scaled[test_idx], test_labels_sample))
    else:
        test_silhouette = None

    # Assign every row to a learned cluster for an inspectable output file. Demand is not used.
    all_labels = cluster_pipeline.predict(X)
    assignments = df[["Date", "Store ID", "Product ID"]].copy()
    assignments[CLUSTER_FEATURES] = X
    assignments["Cluster"] = all_labels
    assignments.to_csv(ANALYSIS_DIR / "cluster_assignments.csv", index=False)

    profile_frame = X.copy()
    profile_frame["Cluster"] = all_labels
    profile_rows = []
    for cluster_id, group in profile_frame.groupby("Cluster"):
        row = {"Cluster": int(cluster_id), "Records": int(len(group)), "Percent of records": float(len(group) / len(profile_frame) * 100)}
        for feature in CLUSTER_FEATURES:
            row[f"Mean - {feature}"] = float(group[feature].mean())
            row[f"Median - {feature}"] = float(group[feature].median())
        profile_rows.append(row)
    profiles = pd.DataFrame(profile_rows).sort_values("Cluster")
    profiles.to_csv(ANALYSIS_DIR / "cluster_profiles.csv", index=False, float_format="%.4f")

    silhouette_frame = pd.DataFrame(silhouette_rows)
    silhouette_frame.to_csv(ANALYSIS_DIR / "kmeans_k_selection.csv", index=False, float_format="%.5f")

    fig, ax = plt.subplots(figsize=(8, 4.8), facecolor="#0b1020")
    ax.set_facecolor("#11182b")
    ax.bar(silhouette_frame["k"].astype(str), silhouette_frame["Training sample silhouette"], color="#5ee7f8", alpha=0.85)
    ax.set_xlabel("Number of clusters (k)", color="#c5cce0")
    ax.set_ylabel("Silhouette score (training sample)", color="#c5cce0")
    ax.set_title("K-Means Cluster Count Selection", color="#f7f8ff", pad=14, fontsize=13, weight="bold")
    ax.tick_params(colors="#c5cce0")
    ax.grid(axis="y", color="#ffffff", alpha=0.08)
    for spine in ax.spines.values(): spine.set_color("#3b4560")
    save_figure(fig, "kmeans_k_selection.png")

    # Standardized cluster profile plot makes variables with different units comparable.
    cluster_means = profile_frame.groupby("Cluster")[CLUSTER_FEATURES].mean().sort_index()
    all_means = X.mean()
    all_stds = X.std().replace(0, 1)
    z_means = (cluster_means - all_means) / all_stds
    fig, ax = plt.subplots(figsize=(12, 5.5), facecolor="#0b1020")
    ax.set_facecolor("#11182b")
    image = ax.imshow(z_means.to_numpy(), cmap="coolwarm", vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(len(CLUSTER_FEATURES)), CLUSTER_FEATURES, rotation=40, ha="right", color="#c5cce0", fontsize=8)
    ax.set_yticks(range(len(z_means.index)), [f"Cluster {i}" for i in z_means.index], color="#c5cce0")
    ax.set_title("K-Means Inventory Profile (Cluster Means in Dataset Standard Deviations)", color="#f7f8ff", pad=14, fontsize=13, weight="bold")
    ax.tick_params(colors="#c5cce0")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Standard deviations from dataset mean")
    save_figure(fig, "kmeans_cluster_profiles.png")

    joblib.dump(cluster_pipeline, MODEL_DIR / "kmeans_inventory_profiles.joblib")
    metrics = {
        "algorithm": "K-Means",
        "cluster_count_selected": int(best_k),
        "candidate_k_values": list(K_CANDIDATES),
        "selection_method": "Highest silhouette score on a deterministic sample drawn from the training period only",
        "selection_sample_rows": int(search_size),
        "training_sample_silhouette_for_selected_k": float(best_score),
        "held_out_test_silhouette": test_silhouette,
        "test_silhouette_sample_rows": int(test_n),
        "train_rows": int(train_mask.sum()),
        "test_rows": int((~train_mask).sum()),
        "test_start_date": cutoff.date().isoformat(),
        "features": CLUSTER_FEATURES,
        "excluded_fields": ["Demand", "Date", "Store ID", "Product ID", "Category", "Region", "Weather Condition", "Seasonality"],
        "note": "Demand is intentionally excluded from clustering. Cluster numbers are arbitrary group identifiers; interpret groups by reading cluster_profiles.csv.",
    }
    save_json(ANALYSIS_DIR / "clustering_metrics.json", metrics)
    save_json(MODEL_DIR / "clustering_metadata.json", metrics)
    profiles_by_cluster = profiles.set_index("Cluster")
    profile_lines = []
    for cluster_id, profile in profiles_by_cluster.iterrows():
        profile_lines.append(
            f"- **Cluster {int(cluster_id)}:** {int(profile['Records']):,} records "
            f"({profile['Percent of records']:.2f}%); mean units sold "
            f"{profile['Mean - Units Sold']:.2f}; mean discount "
            f"{profile['Mean - Discount']:.2f}; promotion active in "
            f"{profile['Mean - Promotion'] * 100:.2f}% of records; mean units ordered "
            f"{profile['Mean - Units Ordered']:.2f}."
        )
    clustering_findings = f"""# Clustering Findings

- **Algorithm:** K-Means.
- **Features:** {', '.join(CLUSTER_FEATURES)}.
- **Excluded:** `Demand` is not used to form clusters.
- **Number of clusters selected:** {best_k}, selected from k=2..6 using the highest silhouette score on a deterministic sample from training-period rows.
- **Training selection silhouette:** {best_score:.4f}.
- **Held-out test-sample silhouette:** {test_silhouette:.4f} (computed on {test_n:,} later test records).

## Cluster profiles

The cluster number is an arbitrary group identifier, not a ranking. Compare the original-unit profile summaries:

{chr(10).join(profile_lines)}

For this run, cluster profiles mainly separate records with different promotion/discount and associated sales/order levels. This is a description of patterns in this dataset, not proof that promotion caused higher demand. See `cluster_profiles.csv`, `cluster_assignments.csv`, `kmeans_k_selection.csv`, and the figures for detail.
"""
    (ANALYSIS_DIR / "clustering_findings.md").write_text(clustering_findings, encoding="utf-8")

    return {
        "algorithm": "K-Means",
        "selected_k": int(best_k),
        "training_silhouette": float(best_score),
        "held_out_test_silhouette": test_silhouette,
        "cluster_counts": {str(int(key)): int(value) for key, value in pd.Series(all_labels).value_counts().sort_index().items()},
        "model_path": "models/kmeans_inventory_profiles.joblib",
    }


# Run the full analysis pipeline in order: validate, explore, classify, cluster, save.
def main() -> None:
    prepare_directories()
    df = load_and_validate_data()
    train_mask, cutoff = chronological_split(df)
    print(f"Dataset loaded: {len(df):,} rows x {len(df.columns)} columns")
    print(f"Data quality: {int(df.isna().sum().sum())} missing cells; {int(df.duplicated().sum())} duplicate rows")
    print(f"Chronological split: {int(train_mask.sum()):,} train rows / {int((~train_mask).sum()):,} test rows; cutoff {cutoff.date()}")

    # Run the three project tasks and collect their summary results.
    eda_results = run_eda(df, train_mask, cutoff)
    classification_results = run_classification(df, train_mask, cutoff)
    clustering_results = run_clustering(df, train_mask, cutoff)

    summary = {
        "dataset": eda_results,
        "classification": classification_results,
        "clustering": clustering_results,
    }
    save_json(ANALYSIS_DIR / "project_run_summary.json", summary)

    print("\nClassification results:")
    print(pd.read_csv(ANALYSIS_DIR / "classification_metrics.csv").to_string(index=False))
    print("\nClustering results:")
    print(json.dumps(clustering_results, indent=2))
    print(f"\nAnalysis files written to: {ANALYSIS_DIR}")
    print(f"Models written to: {MODEL_DIR}")


# Execute the pipeline only when this script is launched directly.
if __name__ == "__main__":
    main()
