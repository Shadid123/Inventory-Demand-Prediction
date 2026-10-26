# Exploratory Data Analysis Findings

## Dataset checks

- **Rows:** 76,000
- **Columns:** 16
- **Date range:** 2022-01-01 to 2024-01-30
- **Missing cells:** 0
- **Exact duplicate rows:** 0
- **Chronological split:** 60,800 earlier records for training and 15,200 later records for testing; test starts 2023-09-01.

No rows were silently removed during this analysis. Missing values and duplicate rows were checked and counted explicitly.

## Demand patterns

- Recorded demand has a mean of **104.32** units and a median of **100.00** units; the range is **4–430** units.
- The category with the highest mean demand is **Groceries** (120.98 units on average in this dataset).
- The category with the lowest mean demand is **Furniture** (73.58 units on average in this dataset).
- The month with the highest mean demand is **2023-08** (120.11 units); the lowest is **2023-02** (71.52 units).
- The strongest absolute numeric correlation with `Demand` is **Units Sold** (Pearson correlation 0.833). Correlation describes association, not proof of causation.
- Mean demand was **95.03** units with no promotion and **123.27** units when promotion was active. This comparison is descriptive and does not isolate promotion as the cause.

## Figures and output files

- `figures/demand_distribution.png` — distribution of the target variable
- `figures/demand_by_category.png` — demand distribution by product category
- `figures/monthly_average_demand.png` — monthly average recorded demand
- `figures/numeric_correlation_heatmap.png` — correlations between numeric fields
- `figures/demand_by_promotion.png` — demand distribution by promotion status
- `dataset_overview.csv`, `numeric_summary.csv`, `categorical_distributions.csv`, `correlations_with_demand.csv`, `demand_by_category.csv`, `monthly_demand_summary.csv`, and `promotion_demand_summary.csv` — supporting tables

## Interpretation caution

`Units Sold` is strongly associated with the recorded `Demand` in this dataset. The regression and classification tasks therefore describe patterns in the supplied records. A real future-demand forecast must ensure every input represents information already available at the forecast time; using sales from the same period as the target can create target leakage for a true forecast.
