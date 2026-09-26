from pathlib import Path
import json

import joblib
import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

app = Flask(__name__)

MODEL_FILES = {
    "Linear Regression": MODEL_DIR / "linear_regression.joblib",
    "Decision Tree": MODEL_DIR / "decision_tree.joblib",
    "Random Forest": MODEL_DIR / "random_forest.joblib",
}

models = {name: joblib.load(path) for name, path in MODEL_FILES.items()}
metadata = json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))
comparison = pd.read_csv(MODEL_DIR / "model_comparison.csv")
comparison_records = comparison.to_dict(orient="records")

best_row = comparison.sort_values("R2", ascending=False).iloc[0]
evaluation_winner = str(best_row["Model"])


def make_input_row(form):
    date = pd.to_datetime(form["date"], errors="raise")

    row = {
        "Store ID": form["store_id"],
        "Product ID": form["product_id"],
        "Category": form["category"],
        "Region": form["region"],
        "Inventory Level": float(form["inventory_level"]),
        "Units Sold": float(form["units_sold"]),
        "Units Ordered": float(form["units_ordered"]),
        "Price": float(form["price"]),
        "Discount": float(form["discount"]),
        "Weather Condition": form["weather_condition"],
        "Promotion": int(form["promotion"]),
        "Competitor Pricing": float(form["competitor_pricing"]),
        "Seasonality": form["seasonality"],
        "Epidemic": int(form["epidemic"]),
        "Year": date.year,
        "Month": date.month,
        "Day": date.day,
        "DayOfWeek": date.dayofweek,
        "DayOfYear": date.dayofyear,
    }
    return pd.DataFrame([row])


@app.route("/", methods=["GET", "POST"])
def index():
    predictions = None
    recommendation = None
    error = None

    defaults = {
        "date": "2024-01-30",
        "store_id": metadata["available_values"]["Store ID"][0],
        "product_id": metadata["available_values"]["Product ID"][0],
        "category": metadata["available_values"]["Category"][0],
        "region": metadata["available_values"]["Region"][0],
        "inventory_level": "100",
        "units_sold": "80",
        "units_ordered": "80",
        "price": "60",
        "discount": "10",
        "weather_condition": metadata["available_values"]["Weather Condition"][0],
        "promotion": "0",
        "competitor_pricing": "65",
        "seasonality": metadata["available_values"]["Seasonality"][0],
        "epidemic": "0",
    }

    if request.method == "POST":
        defaults.update(request.form.to_dict())
        try:
            row = make_input_row(request.form)
            predictions = {
                name: round(float(model.predict(row)[0]), 2)
                for name, model in models.items()
            }

            recommended_demand = predictions[evaluation_winner]
            inventory = float(request.form["inventory_level"])
            reorder_qty = max(0, round(recommended_demand - inventory))

            recommendation = {
                "recommended_demand": recommended_demand,
                "status": "REORDER REQUIRED" if reorder_qty > 0 else "STOCK SUFFICIENT",
                "reorder_qty": reorder_qty,
                "model": evaluation_winner,
            }
        except (ValueError, KeyError) as exc:
            error = f"Please check the input values. Details: {exc}"

    return render_template(
        "index.html",
        predictions=predictions,
        recommendation=recommendation,
        error=error,
        form=defaults,
        values=metadata["available_values"],
        comparison=comparison_records,
        evaluation_winner=evaluation_winner,
    )


if __name__ == "__main__":
    app.run(debug=True)
