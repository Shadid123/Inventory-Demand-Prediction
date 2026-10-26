# Inventory Demand Prediction

A beginner-friendly inventory data-mining project. It explores retail inventory records, builds models that estimate demand, groups similar records, and presents the results in a local web dashboard.

> **Dataset note:** The included Kaggle dataset is explicitly described by its publisher as *synthetic* (computer-generated), not verified records from a real shop. This project is useful for demonstrating data-mining methods, but its results should not be presented as proven real-world retail findings. If your course requires observed real-world data, confirm whether this synthetic dataset is acceptable before submitting it.

## What does this project do?

Imagine a shop that wants to understand how stock, sales, price, discounts and promotions relate to demand. This project uses a historical spreadsheet to explore that information and demonstrate three different kinds of data-mining tasks.

- **Estimate a number (regression):** Linear Regression, Decision Tree Regressor and Random Forest Regressor each estimate the `Demand` value.
- **Predict a category (classification):** a Decision Tree Classifier labels a record as **Low**, **Medium** or **High** demand. These labels are created from the training data; they are not original columns in the dataset.
- **Find similar groups (clustering):** K-Means groups records with similar selected inventory and market values. It does not use `Demand` to create the groups.
- **Explore the data (EDA):** summary tables and charts help us inspect demand values, monthly patterns, product categories, promotions and relationships between numerical columns.
- **Show the results in a website:** a Flask app lets a user enter information and see model predictions, a demand category, a cluster assignment and a simple stock recommendation.

## How the system works

```text
sales_data.csv
     |
     v
Check and prepare the data
     |
     +--> Explore patterns and make charts
     |
     +--> Train 3 regression models --> Compare their errors
     |
     +--> Train a demand classifier --> Evaluate its categories
     |
     +--> Train K-Means --> Inspect similar-record groups
                                      |
                                      v
                           Saved models and results
                                      |
                                      v
                              Flask web dashboard
```

## Dataset

The file is `data/sales_data.csv`. The included file has **76,000 rows and 16 columns**, dated from **January 1, 2022 to January 30, 2024**. The analysis script checks for missing cells and exact duplicate rows; in the supplied file, both checks returned zero.

Important columns include:

| Column | Plain-English meaning |
|---|---|
| `Date` | Date of the record |
| `Store ID` | Store identifier |
| `Product ID` | Product identifier |
| `Category` | Product category |
| `Region` | Store region |
| `Inventory Level` | Recorded stock level |
| `Units Sold` | Number of units sold in the record |
| `Units Ordered` | Number of units ordered |
| `Price` | Product price |
| `Discount` | Discount value recorded in the data |
| `Weather Condition` | Recorded weather category |
| `Promotion` | Promotion indicator (0/1) |
| `Competitor Pricing` | Comparable competitor price |
| `Seasonality` | Season category |
| `Epidemic` | Epidemic indicator (0/1) |
| `Demand` | Numeric demand value to be estimated by the regression models |

**Dataset source:** [Retail Store Inventory and Demand Forecasting — Kaggle](https://www.kaggle.com/dsv/11895299). The source page states that the dataset is synthetic and lists an Apache 2.0 license. Review the source page for the latest attribution and usage terms.

## The methods, explained simply

### 1. Preprocessing

Before the models use the data, the scripts:

- Check the required columns, missing values and duplicate records.
- Extract calendar features such as year, month and weekday from `Date`.
- Convert word-based categories into numerical columns using one-hot encoding where required.
- Standardize selected numeric features for K-Means, because it groups records using distances.
- Split records chronologically: earlier records are used for training and later records for testing.

The supplied dataset has no missing cells, so the scripts do not need to fill missing values for this run.

### 2. Exploratory Data Analysis (EDA)

EDA means looking at the data before interpreting the models. The project creates summaries and charts for demand distribution, monthly demand, demand by category, promotion comparisons, numerical correlations and candidate cluster counts. Results are saved in `analysis/`.

### 3. Regression: estimating demand as a number

Three regression approaches are trained on the same chronological split:

- **Linear Regression:** a simple mathematical baseline for relationships between inputs and demand.
- **Decision Tree Regressor:** learns decision-like splits in the data.
- **Random Forest Regressor:** combines many decision trees to produce a prediction.

Their comparison results are saved in `models/model_comparison.csv`. MAE and RMSE measure prediction error (lower is better); R² compares the model with a mean-based baseline (higher is generally better). Results can change if the models are retrained or the data is changed.

### 4. Classification: assigning a demand level

The original file contains numeric `Demand`, not Low/Medium/High labels. The project creates three labels using cutoffs calculated **from the training period only**, then trains a Decision Tree Classifier. Its report and confusion matrix are saved in `analysis/`.

The classifier excludes `Demand` from its inputs. However, `Units Sold` remains an input because it exists in the supplied record. The dataset alone does not confirm whether that sales value would be known before a future period is forecast, so this should be described as classifying the recorded observation rather than proof of future-demand forecasting.

### 5. Clustering: finding similar inventory records

K-Means groups records using selected numeric inventory and market features, such as stock, units sold, units ordered, price and discount. `Demand` is excluded when forming the clusters. Candidate cluster counts are compared with a silhouette score, and the script saves cluster profiles to `analysis/cluster_profiles.csv`.

Cluster IDs such as 0 and 1 are just labels; one cluster is not automatically better than another. Read the cluster profiles to understand how groups differ.

### 6. Evaluation

Each task is evaluated with metrics suited to its purpose:

| Task | Evaluation |
|---|---|
| Regression | MAE, RMSE and R² |
| Classification | Accuracy, balanced accuracy, precision, recall, F1-score and confusion matrix |
| Clustering | Silhouette scores and cluster-profile summaries |

## Run it on Windows (PowerShell)

These instructions are written for Windows. They deliberately use the virtual environment's Python executable directly, so **you do not need to activate the environment**. This also avoids the PowerShell execution-policy error that can block `Activate.ps1`.

### Step 1 — Open the project folder

Extract or clone the repository, then open the folder containing `app.py`, `train_model.py` and `requirements.txt` in VS Code. Open **Terminal → New Terminal**. Make sure the terminal is in that project folder.

### Step 2 — Create a private Python environment

```powershell
python -m venv .venv
```

A `.venv` folder is created. It keeps this project's packages separate from other Python projects on your computer.

### Step 3 — Install the packages

Run these commands in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The installation can take a few minutes, depending on your connection. The project pins scikit-learn to `1.9.1` so the project can use one consistent version when training and loading its saved models.

### Step 4 — Train the three regression models

```powershell
.\.venv\Scripts\python.exe train_model.py
```

This trains Linear Regression, Decision Tree Regressor and Random Forest Regressor, evaluates them on later-dated records, and saves the fitted pipelines in `models/`.

### Step 5 — Run EDA, classification and clustering

```powershell
.\.venv\Scripts\python.exe run_inventory_analysis.py
```

This creates the analysis tables and charts, trains the classification and K-Means models, and saves their results. Running both training scripts after installing the dependencies also ensures that all saved model files are created using the same local scikit-learn version.

### Step 6 — Start the website

```powershell
.\.venv\Scripts\python.exe app.py
```

Wait for Flask to show a local address, normally:

```text
http://127.0.0.1:5000
```

Open that address in Chrome or Edge. **Keep the terminal open while using the website.** To stop the server, return to the terminal and press `Ctrl+C`.

### Step 7 — Try a prediction

1. Fill in the product, store, date, stock and market information.
2. Select **Predict Demand**.
3. Review the three numerical demand estimates.
4. Review the predicted Low/Medium/High demand label and K-Means cluster assignment.
5. Read the suggested stock action. It is a simplified comparison of predicted demand with the entered stock level, not a complete purchasing plan.

## Useful output files

| File or folder | What it contains |
|---|---|
| `analysis/eda_findings.md` | Plain-language EDA observations |
| `analysis/figures/` | EDA and evaluation charts |
| `analysis/classification_report.csv` | Per-class classification scores |
| `analysis/classification_confusion_matrix.csv` | Correct and incorrect category predictions |
| `analysis/classification_metrics.csv` | Overall classification metrics and baseline |
| `analysis/cluster_profiles.csv` | Summary of the groups found by K-Means |
| `analysis/clustering_metrics.json` | Cluster selection and silhouette scores |
| `analysis/project_run_summary.json` | Summary of the latest EDA/classification/clustering run |
| `models/model_comparison.csv` | Regression model comparison |
| `models/` | Saved model files used by the website |

## Troubleshooting

**PowerShell says running scripts is disabled.** You do not need to run `.venv\Scripts\Activate.ps1`. Continue using the commands above that start with `.\.venv\Scripts\python.exe`.

**A package is missing.** From the project folder, run:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**You see a scikit-learn `InconsistentVersionWarning`.** Stop the app with `Ctrl+C`, install the listed requirements, then rerun both `train_model.py` and `run_inventory_analysis.py` before starting `app.py`. Do not ignore a version mismatch if you are reporting model results.

**Jinja template text appears on the page or the styling is missing.** Do not open `templates/index.html` directly or use VS Code Live Server. Start the Flask app with `app.py`, then open `http://127.0.0.1:5000`.

**The dataset cannot be found.** Confirm the file exists at `data/sales_data.csv` relative to the folder containing `app.py`.

## Project structure

```text
inventory-demand-prediction/
├── analysis/
│   ├── figures/
│   ├── eda_findings.md
│   ├── classification_report.csv
│   ├── classification_metrics.csv
│   ├── classification_confusion_matrix.csv
│   ├── cluster_profiles.csv
│   └── project_run_summary.json
├── data/
│   └── sales_data.csv
├── models/
│   ├── linear_regression.joblib
│   ├── decision_tree.joblib
│   ├── random_forest.joblib
│   ├── demand_level_classifier.joblib
│   └── kmeans_inventory_profiles.joblib
├── static/
│   └── style.css
├── templates/
│   └── index.html
├── app.py
├── train_model.py
├── run_inventory_analysis.py
├── requirements.txt
├── DESIGN_SYSTEM.md
└── README.md
```

## Limitations to keep in mind

- The source dataset is synthetic. Patterns found in it may not represent an actual retailer.
- `Units Sold` is strongly associated with `Demand` in the supplied file. The timing of the columns must be understood before describing the app as a genuine future-forecasting system.
- The stock recommendation is intentionally simple. A real inventory decision would also consider supplier lead time, safety stock, existing purchase orders and business constraints.
- Model scores are measurements on this dataset and test split; they are not a guarantee of performance on new real-world data.

## Design

The dashboard uses a dark glassmorphism style with translucent panels, violet/cyan accents and responsive layouts. See [`DESIGN_SYSTEM.md`](DESIGN_SYSTEM.md) for the visual design tokens and interaction details.
## Reading the code

The main source files include beginner-friendly comments explaining what each section does and why it exists:

- `app.py` — connects the browser form to the saved models and displays predictions.
- `train_model.py` — trains and evaluates the three demand-regression models.
- `run_inventory_analysis.py` — creates EDA outputs, trains the demand classifier, and runs K-Means clustering.
- `templates/index.html` — defines the page structure and form fields.
- `static/style.css` — controls the visual appearance and responsive layout.

Comments focus on meaningful statements, model decisions, and groups of related HTML/CSS rules. They do not repeat obvious punctuation or every individual CSS property, because that would make the files harder to read rather than easier.

