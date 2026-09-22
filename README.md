# Inventory Management — Data Mining Project

## Project objective

This project predicts **inventory demand** from retail data and compares three data-mining / machine-learning regression models:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor

A Flask web interface lets a user enter inventory and sales information and see all three predictions.

## Dataset

Place the supplied `sales_data.csv` file in:

```text
data/sales_data.csv
```

The dataset contains 76,000 records and 16 columns. `Demand` is the prediction target.

## Model evaluation

The project uses a **chronological 80/20 train-test split** rather than a random split. This is more appropriate for inventory demand because it avoids training on later dates and testing on earlier dates.

On the supplied dataset, the current run produced approximately:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Random Forest | 12.18 | 15.93 | 0.870 |
| Decision Tree | 14.66 | 20.00 | 0.794 |
| Linear Regression | 15.61 | 20.82 | 0.777 |

Random Forest is used by the interface for the inventory recommendation because it had the lowest test RMSE in this run.

> Important interpretation: `Units Sold` is a very strong predictor in this dataset. For a real future-forecasting deployment, that value must mean recent/known sales rather than sales from the same future period being predicted.

## How a non-technical user can run it

### 1. Install Python

Install Python 3.10 or newer from the official Python website.

During installation, enable **Add Python to PATH** if that option is shown.

### 2. Open the project folder

Open this folder in File Explorer or VS Code.

### 3. Open a terminal in the project folder

On Windows, right-click inside the folder and choose **Open in Terminal**.

### 4. Install the required packages

Run:

```bash
pip install -r requirements.txt
```

If `pip` does not work, try:

```bash
python -m pip install -r requirements.txt
```

### 5. Start the web application

Pre-trained model files are already included in the project, so you can run the interface directly:

```bash
python app.py
```

You only need to run the training script again when you change the dataset or want to retrain the models:

```bash
python train_model.py
```

You should see a local address such as:

```text
http://127.0.0.1:5000
```

Open that address in Chrome or Edge.

### 7. Use the interface

Fill in the inventory and sales information and click:

**Predict Demand**

The page will show:

- Linear Regression prediction
- Decision Tree prediction
- Random Forest prediction
- Expected demand
- Current stock
- Reorder status
- Suggested reorder quantity

## Project structure

```text
inventory_management_project/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
│
├── data/
│   └── sales_data.csv
│
├── models/
│   ├── linear_regression.joblib
│   ├── decision_tree.joblib
│   ├── random_forest.joblib
│   ├── model_comparison.csv
│   └── metadata.json
│
├── templates/
│   └── index.html
│
└── static/
    └── style.css
```

## Presentation points

Explain these topics in order:

1. Inventory-management problem
2. Dataset and features
3. Data preprocessing
4. Why demand prediction is useful
5. Linear Regression
6. Decision Tree
7. Random Forest
8. MAE, RMSE and R²
9. Model comparison
10. Flask interface
11. Example prediction
12. Conclusion and future improvements

## Future improvements

- Add historical lag features such as previous-day or previous-week sales.
- Add reorder-point and safety-stock calculations.
- Add charts for demand trends.
- Store predictions in a database.
- Add authentication for inventory managers.
- Deploy the Flask application online.
