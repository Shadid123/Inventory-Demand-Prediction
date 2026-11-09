from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Inventory Demand Prediction",
    page_icon="📦",
    layout="wide",
)

# Resolve paths relative to this file, so deployment does not depend on the
# current working directory.
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
ANALYSIS_DIR = BASE_DIR / "analysis"

CLUSTER_FEATURES = [
    "Inventory Level",
    "Units Sold",
    "Units Ordered",
    "Price",
    "Discount",
    "Competitor Pricing",
    "Promotion",
    "Epidemic",
]

MODEL_FILES = {
    "Linear Regression": MODEL_DIR / "linear_regression.joblib",
    "Decision Tree": MODEL_DIR / "decision_tree.joblib",
    "Random Forest": MODEL_DIR / "random_forest.joblib",
}


@st.cache_resource
def load_project_assets():
    """Load the trained models and supporting files once per Streamlit session."""
    required_files = list(MODEL_FILES.values()) + [
        MODEL_DIR / "metadata.json",
        MODEL_DIR / "model_comparison.csv",
        MODEL_DIR / "demand_level_classifier.joblib",
        MODEL_DIR / "classification_metadata.json",
        MODEL_DIR / "kmeans_inventory_profiles.joblib",
        MODEL_DIR / "clustering_metadata.json",
        ANALYSIS_DIR / "cluster_profiles.csv",
    ]
    missing = [str(path.relative_to(BASE_DIR)) for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "The following project files are missing: " + ", ".join(missing)
        )

    models = {name: joblib.load(path) for name, path in MODEL_FILES.items()}
    metadata = json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))
    comparison = pd.read_csv(MODEL_DIR / "model_comparison.csv")
    classification_model = joblib.load(MODEL_DIR / "demand_level_classifier.joblib")
    classification_metadata = json.loads(
        (MODEL_DIR / "classification_metadata.json").read_text(encoding="utf-8")
    )
    clustering_model = joblib.load(MODEL_DIR / "kmeans_inventory_profiles.joblib")
    clustering_metadata = json.loads(
        (MODEL_DIR / "clustering_metadata.json").read_text(encoding="utf-8")
    )
    cluster_profiles = pd.read_csv(ANALYSIS_DIR / "cluster_profiles.csv")

    best_row = comparison.sort_values("R2", ascending=False).iloc[0]
    evaluation_winner = str(best_row["Model"])

    return (
        models,
        metadata,
        comparison,
        classification_model,
        classification_metadata,
        clustering_model,
        clustering_metadata,
        cluster_profiles,
        evaluation_winner,
    )


def get_options(metadata, column_name, fallback):
    values = metadata.get("available_values", {}).get(column_name, [])
    return values if values else fallback


def make_input_row(values):
    """Build one input row using the same column names as the Flask app."""
    selected_date = values["date"]
    row = {
        "Store ID": values["store_id"],
        "Product ID": values["product_id"],
        "Category": values["category"],
        "Region": values["region"],
        "Inventory Level": float(values["inventory_level"]),
        "Units Sold": float(values["units_sold"]),
        "Units Ordered": float(values["units_ordered"]),
        "Price": float(values["price"]),
        "Discount": float(values["discount"]),
        "Weather Condition": values["weather_condition"],
        "Promotion": int(values["promotion"]),
        "Competitor Pricing": float(values["competitor_pricing"]),
        "Seasonality": values["seasonality"],
        "Epidemic": int(values["epidemic"]),
        "Year": selected_date.year,
        "Month": selected_date.month,
        "Day": selected_date.day,
        "DayOfWeek": selected_date.weekday(),
        "DayOfYear": selected_date.timetuple().tm_yday,
    }
    return pd.DataFrame([row])


st.title("📦 Inventory Demand Prediction")
st.caption(
    "Estimate product demand, identify the demand level, and review an inventory reorder suggestion."
)

try:
    (
        models,
        metadata,
        comparison,
        classification_model,
        classification_metadata,
        clustering_model,
        clustering_metadata,
        cluster_profiles,
        evaluation_winner,
    ) = load_project_assets()
except Exception as exc:
    st.error("The application could not load its project files.")
    st.exception(exc)
    st.info(
        "Check that the folders named 'models' and 'analysis' are beside "
        "streamlit_app.py and contain the files listed in the error."
    )
    st.stop()

available = metadata.get("available_values", {})

with st.expander("About this project", expanded=False):
    st.write(
        "This application uses three trained regression models to estimate demand. "
        "A classification model predicts a demand category, and a K-Means model "
        "assigns the input to an inventory profile."
    )
    st.write(f"Regression model selected for the reorder suggestion: **{evaluation_winner}**")

st.subheader("Enter inventory and sales details")
st.write("The fields below start with sample values. Change them to test another scenario.")

with st.form("prediction_form"):
    date_col, store_col, product_col = st.columns(3)
    with date_col:
        selected_date = st.date_input("Date", value=pd.Timestamp("2024-01-30").date())
    with store_col:
        store_id = st.selectbox(
            "Store ID", get_options(metadata, "Store ID", ["Store 1"])
        )
    with product_col:
        product_id = st.selectbox(
            "Product ID", get_options(metadata, "Product ID", ["Product 1"])
        )

    category_col, region_col, weather_col = st.columns(3)
    with category_col:
        category = st.selectbox(
            "Category", get_options(metadata, "Category", ["General"])
        )
    with region_col:
        region = st.selectbox("Region", get_options(metadata, "Region", ["Unknown"]))
    with weather_col:
        weather_condition = st.selectbox(
            "Weather Condition",
            get_options(metadata, "Weather Condition", ["Normal"]),
        )

    inventory_col, sold_col, ordered_col = st.columns(3)
    with inventory_col:
        inventory_level = st.number_input("Inventory Level", min_value=0.0, value=100.0)
    with sold_col:
        units_sold = st.number_input("Units Sold", min_value=0.0, value=80.0)
    with ordered_col:
        units_ordered = st.number_input("Units Ordered", min_value=0.0, value=80.0)

    price_col, discount_col, competitor_col = st.columns(3)
    with price_col:
        price = st.number_input("Price", min_value=0.0, value=60.0)
    with discount_col:
        discount = st.number_input("Discount", min_value=0.0, value=10.0)
    with competitor_col:
        competitor_pricing = st.number_input(
            "Competitor Pricing", min_value=0.0, value=65.0
        )

    season_col, promotion_col, epidemic_col = st.columns(3)
    with season_col:
        seasonality = st.selectbox(
            "Seasonality", get_options(metadata, "Seasonality", ["Regular"])
        )
    with promotion_col:
        promotion = st.selectbox("Promotion", [0, 1], format_func=lambda x: "Yes" if x else "No")
    with epidemic_col:
        epidemic = st.selectbox("Epidemic", [0, 1], format_func=lambda x: "Yes" if x else "No")

    submitted = st.form_submit_button("Predict demand", type="primary", use_container_width=True)

if submitted:
    form_values = {
        "date": selected_date,
        "store_id": store_id,
        "product_id": product_id,
        "category": category,
        "region": region,
        "inventory_level": inventory_level,
        "units_sold": units_sold,
        "units_ordered": units_ordered,
        "price": price,
        "discount": discount,
        "weather_condition": weather_condition,
        "promotion": promotion,
        "competitor_pricing": competitor_pricing,
        "seasonality": seasonality,
        "epidemic": epidemic,
    }

    try:
        input_row = make_input_row(form_values)

        predictions = {
            name: round(float(model.predict(input_row)[0]), 2)
            for name, model in models.items()
        }

        predicted_level = str(classification_model.predict(input_row)[0])
        class_rule = classification_metadata["thresholds"]["rules"].get(
            predicted_level, "See the classification metadata for this category."
        )

        assigned_cluster = int(clustering_model.predict(input_row[CLUSTER_FEATURES])[0])
        profile_matches = cluster_profiles.loc[
            cluster_profiles["Cluster"] == assigned_cluster
        ]
        profile = profile_matches.iloc[0] if not profile_matches.empty else None

        recommended_demand = predictions[evaluation_winner]
        reorder_qty = max(0, round(recommended_demand - float(inventory_level)))

        st.divider()
        st.subheader("Prediction results")
        prediction_cols = st.columns(len(predictions))
        for column, (model_name, predicted_demand) in zip(prediction_cols, predictions.items()):
            column.metric(model_name, f"{predicted_demand:,.2f} units")

        result_col1, result_col2 = st.columns(2)
        with result_col1:
            st.markdown("#### Demand category")
            st.metric("Predicted demand level", predicted_level)
            st.caption(str(class_rule))
        with result_col2:
            st.markdown("#### Inventory recommendation")
            if reorder_qty > 0:
                st.error("REORDER REQUIRED")
            else:
                st.success("STOCK SUFFICIENT")
            st.metric("Suggested reorder quantity", f"{reorder_qty:,} units")
            st.caption(
                f"Based on {evaluation_winner}: estimated demand "
                f"{recommended_demand:,.2f} units."
            )

        st.markdown("#### Inventory profile")
        st.write(f"Assigned to **Cluster {assigned_cluster + 1}** of {clustering_metadata['cluster_count_selected']} clusters.")
        if profile is not None:
            profile_cols = st.columns(4)
            profile_cols[0].metric("Records in profile", f"{int(profile['Records']):,}")
            profile_cols[1].metric("Mean inventory", f"{float(profile['Mean - Inventory Level']):,.1f}")
            profile_cols[2].metric("Mean units sold", f"{float(profile['Mean - Units Sold']):,.1f}")
            profile_cols[3].metric("Mean discount", f"{float(profile['Mean - Discount']):,.1f}")
            st.caption(
                f"Promotion rate in this profile: "
                f"{float(profile['Mean - Promotion']) * 100:.1f}%"
            )
        else:
            st.info("No saved profile summary was found for this cluster.")

        st.divider()
        st.subheader("Model comparison")
        display_comparison = comparison.copy()
        st.dataframe(display_comparison, use_container_width=True, hide_index=True)
        if "Model" in comparison.columns and "R2" in comparison.columns:
            chart_data = comparison.set_index("Model")["R2"]
            st.bar_chart(chart_data)

        with st.expander("Classification evaluation metrics"):
            st.write("Model metrics")
            st.json(classification_metadata.get("metrics", {}))
            st.write("Baseline metrics")
            st.json(classification_metadata.get("baseline_metrics", {}))

        with st.expander("Clustering evaluation details"):
            st.json(clustering_metadata)

    except (ValueError, KeyError, TypeError, IndexError) as exc:
        st.error(f"Please check the input values or project metadata. Details: {exc}")
    except Exception as exc:
        st.error("An unexpected error occurred while generating the prediction.")
        st.exception(exc)
else:
    st.info("Enter or keep the sample values above, then select **Predict demand**.")

st.caption("Inventory Demand Prediction · Machine-learning project demonstration")
