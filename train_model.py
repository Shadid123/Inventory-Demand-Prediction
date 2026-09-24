
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "sales_data.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

TARGET = "Demand"
MODEL_SPECS = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(
        max_depth=18, min_samples_leaf=3, random_state=42
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=50, max_depth=15, min_samples_leaf=3,
        random_state=42, n_jobs=-1
    ),
}

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="raise")
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["DayOfYear"] = df["Date"].dt.dayofyear
    return df.drop(columns=["Date"])

def main():
    df = pd.read_csv(DATA_PATH)

    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' was not found.")

    features = make_features(df)
    X = features.drop(columns=[TARGET])
    y = features[TARGET]

    # Chronological split: train on earlier dates, test on later dates.
    original_dates = pd.to_datetime(df["Date"])
    unique_dates = np.sort(original_dates.unique())
    cutoff = unique_dates[int(len(unique_dates) * 0.80)]

    train_mask = original_dates < cutoff
    X_train, X_test = X.loc[train_mask], X.loc[~train_mask]
    y_train, y_test = y.loc[train_mask], y.loc[~train_mask]

    categorical_cols = [c for c in X.columns if X[c].dtype == "object"]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    results = []

    for name, estimator in MODEL_SPECS.items():
        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
                ("num", "passthrough", numeric_cols),
            ]
        )

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )

        pipeline.fit(X_train, y_train)
        prediction = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, prediction)
        rmse = mean_squared_error(y_test, prediction) ** 0.5
        r2 = r2_score(y_test, prediction)

        filename = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(pipeline, MODEL_DIR / filename)

        results.append(
            {
                "Model": name,
                "MAE": round(mae, 4),
                "RMSE": round(rmse, 4),
                "R2": round(r2, 4),
            }
        )

    results_df = pd.DataFrame(results).sort_values("RMSE")
    results_df.to_csv(MODEL_DIR / "model_comparison.csv", index=False)

    metadata = {
        "target": TARGET,
        "train_rows": int(train_mask.sum()),
        "test_rows": int((~train_mask).sum()),
        "test_start_date": str(pd.Timestamp(cutoff).date()),
        "date_range": {
            "start": str(original_dates.min().date()),
            "end": str(original_dates.max().date()),
        },
        "categorical_columns": categorical_cols,
        "numeric_columns": numeric_cols,
        "recommended_model": "Random Forest",
        "available_values": {
            "Store ID": sorted(df["Store ID"].unique().tolist()),
            "Product ID": sorted(df["Product ID"].unique().tolist()),
            "Category": sorted(df["Category"].unique().tolist()),
            "Region": sorted(df["Region"].unique().tolist()),
            "Weather Condition": sorted(df["Weather Condition"].unique().tolist()),
            "Seasonality": sorted(df["Seasonality"].unique().tolist()),
        },
    }
    (MODEL_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nModel comparison:")
    print(results_df.to_string(index=False))
    print(f"\nModels saved to: {MODEL_DIR}")
    print(f"Chronological test set starts: {pd.Timestamp(cutoff).date()}")

if __name__ == "__main__":
    main()
