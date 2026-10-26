# Methodology and Reproducibility Notes

## 1. Data checks and preprocessing

`run_inventory_analysis.py` verifies that the expected 16 columns exist and parses `Date` as a date. On the supplied file, there are 76,000 rows, no missing cells, and no exact duplicate rows; therefore no rows are dropped. The tabular summaries in `analysis/` record the checks and the observed distributions.

For regression/classification, `Date` is transformed into `Year`, `Month`, `Day`, `DayOfWeek`, and `DayOfYear`. Categorical predictors are one-hot encoded and unknown categories are ignored. The regression pipeline passes numeric predictors through as-is because the supplied file has no missing cells. The classifier pipeline includes median imputation for numeric predictors and most-frequent imputation for categorical predictors. For K-Means, the numeric features are median-imputed and standardized because they use different units and ranges.

## 2. Train/test design

The workflow sorts unique dates and places all rows before 2023-09-01 in training, with rows on/after 2023-09-01 in testing. This produces 60,800 training rows and 15,200 test rows. The split is chronological so that later observations are held out. The same boundary is used across the main regression, classification, and clustering workflows.

## 3. Classification target and features

The task is **three-class classification of recorded demand level**. The lower and upper cutoffs are the 33⅓rd and 66⅔rd percentiles of training-period `Demand`, calculated using training rows only. In the supplied data those cutoffs are 82 and 121 units. Labels are defined as Low (`Demand <= 82`), Medium (`82 < Demand <= 121`), and High (`Demand > 121`). A Decision Tree Classifier is trained on the input columns other than `Demand`.

This is a derived classification task because the original target is numeric. Demand labels for the test set are created using the fixed training cutoffs, not quantiles recomputed from the test period. Evaluation includes accuracy, balanced accuracy, macro precision/recall/F1, weighted F1, per-class report, confusion matrix, and a most-frequent-class baseline.

## 4. Clustering inputs and model selection

K-Means is trained on `Inventory Level`, `Units Sold`, `Units Ordered`, `Price`, `Discount`, `Competitor Pricing`, `Promotion`, and `Epidemic`. `Demand` is excluded so that the clustering is based on inventory and market signals rather than the target. IDs, dates, and nominal category/weather/season columns are also excluded from this numeric-only clustering analysis.

Candidate `k` values 2, 3, 4, 5, and 6 are compared with silhouette scores calculated on a deterministic sample from the training period. The candidate with the highest sample silhouette score is chosen. The final K-Means pipeline is fitted on all training rows and evaluated on a sample from the later test period. Cluster profiles report original-unit means/medians to make clusters interpretable. Cluster numbers are arbitrary and must not be described as a ranking.

## 5. Interpretation limitation

In the supplied dataset, `Units Sold` has a Pearson correlation of approximately 0.833 with `Demand`. This is a strong association, but correlation does not show causation. The supplied schema does not by itself establish whether `Units Sold` and `Units Ordered` are known before the period represented by `Demand`. Therefore, the current project should be described as mining and classifying the supplied recorded observations unless the timing of these features is verified for a future-forecasting use case.
