# Path helps build file locations that work on different computers.
from pathlib import Path
# json reads the small metadata files saved alongside the trained models.
import json

# joblib loads machine-learning models previously saved by the training scripts.
import joblib
# pandas turns the submitted form values into a table with named columns.
import pandas as pd
# Flask runs the local website; render_template builds the page; request reads form submissions.
from flask import Flask, render_template, request

# Find the project folder from this file so paths do not depend on the terminal location.
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
ANALYSIS_DIR = BASE_DIR / "analysis"
# These are the eight columns K-Means was trained to use; Demand is deliberately excluded.
CLUSTER_FEATURES = [
    "Inventory Level", "Units Sold", "Units Ordered", "Price", "Discount",
    "Competitor Pricing", "Promotion", "Epidemic",
]

# Create the Flask application that receives browser requests.
app = Flask(__name__)

MODEL_FILES = {
    "Linear Regression": MODEL_DIR / "linear_regression.joblib",
    "Decision Tree": MODEL_DIR / "decision_tree.joblib",
    "Random Forest": MODEL_DIR / "random_forest.joblib",
}

# Load each trained regression model so all three can predict from the same user input.
models = {name: joblib.load(path) for name, path in MODEL_FILES.items()}
metadata = json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))
comparison = pd.read_csv(MODEL_DIR / "model_comparison.csv")
comparison_records = comparison.to_dict(orient="records")
classification_model = joblib.load(MODEL_DIR / "demand_level_classifier.joblib")
classification_metadata = json.loads(
    (MODEL_DIR / "classification_metadata.json").read_text(encoding="utf-8")
)
clustering_model = joblib.load(MODEL_DIR / "kmeans_inventory_profiles.joblib")
clustering_metadata = json.loads(
    (MODEL_DIR / "clustering_metadata.json").read_text(encoding="utf-8")
)
cluster_profiles = pd.read_csv(ANALYSIS_DIR / "cluster_profiles.csv")

# Pick the model with the highest saved R² score for the inventory recommendation.
best_row = comparison.sort_values("R2", ascending=False).iloc[0]
evaluation_winner = str(best_row["Model"])


# Convert browser form fields into the exact column format expected by the saved models.
def make_input_row(form):
    # Parse the chosen date so calendar features can be extracted reliably.
    date = pd.to_datetime(form["date"], errors="raise")

    # Convert text/numeric form fields into one record with training-compatible column names.
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
    # Models expect a table (even for one prediction), so wrap the record in a DataFrame.
    return pd.DataFrame([row])


# Show the main page on GET; process submitted prediction inputs on POST.
@app.route("/", methods=["GET", "POST"])
def index():
    predictions = None
    recommendation = None
    classification_result = None
    cluster_result = None
    error = None

    # These starter values make the form usable immediately; the visitor can change them.
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

    # Only run the models after the user submits the form.
    if request.method == "POST":
        defaults.update(request.form.to_dict())
        try:
            row = make_input_row(request.form)
            # Ask each regression model for a numeric demand estimate.
            predictions = {
                name: round(float(model.predict(row)[0]), 2)
                for name, model in models.items()
            }

            # The classifier predicts a demand band (Low, Medium, or High), not an exact number.
            predicted_level = str(classification_model.predict(row)[0])
            class_rule = classification_metadata["thresholds"]["rules"][predicted_level]
            classification_result = {
                "label": predicted_level,
                "rule": class_rule,
            }

            # K-Means assigns the record to the closest learned inventory cluster.
            assigned_cluster = int(clustering_model.predict(row[CLUSTER_FEATURES])[0])
            profile_matches = cluster_profiles.loc[cluster_profiles["Cluster"] == assigned_cluster]
            profile = profile_matches.iloc[0] if not profile_matches.empty else None
            cluster_result = {
                "cluster_label": assigned_cluster + 1,
                "records": int(profile["Records"]) if profile is not None else None,
                "mean_inventory": round(float(profile["Mean - Inventory Level"]), 1) if profile is not None else None,
                "mean_units_sold": round(float(profile["Mean - Units Sold"]), 1) if profile is not None else None,
                "mean_discount": round(float(profile["Mean - Discount"]), 1) if profile is not None else None,
                "promotion_rate": round(float(profile["Mean - Promotion"]) * 100, 1) if profile is not None else None,
                "cluster_count": int(clustering_metadata["cluster_count_selected"]),
            }

            # Use the selected comparison winner as the demand estimate for the reorder suggestion.
            recommended_demand = predictions[evaluation_winner]
            inventory = float(request.form["inventory_level"])
            # Suggest only a positive shortfall; never recommend a negative reorder quantity.
            reorder_qty = max(0, round(recommended_demand - inventory))

            recommendation = {
                "recommended_demand": recommended_demand,
                "status": "REORDER REQUIRED" if reorder_qty > 0 else "STOCK SUFFICIENT",
                "reorder_qty": reorder_qty,
                "model": evaluation_winner,
            }
        # Show a helpful message if an input cannot be converted or a required field is missing.
        except (ValueError, KeyError) as exc:
            error = f"Please check the input values. Details: {exc}"

    # Send the input values, predictions, metrics, and messages to the HTML template.
    return render_template(
        "index.html",
        predictions=predictions,
        recommendation=recommendation,
        classification_result=classification_result,
        cluster_result=cluster_result,
        error=error,
        form=defaults,
        values=metadata["available_values"],
        comparison=comparison_records,
        evaluation_winner=evaluation_winner,
        classification_metrics=classification_metadata["metrics"],
        classification_baseline_metrics=classification_metadata["baseline_metrics"],
        clustering_metrics=clustering_metadata,
    )


# Start the local development server when this file is run directly.
if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)