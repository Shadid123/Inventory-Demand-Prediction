# Classification Findings

- **Algorithm:** Decision Tree Classifier.
- **Task:** classify each recorded observation as Low, Medium or High demand.
- **Training-only cutoffs:** Low is `Demand <= 82`; Medium is `82 < Demand <= 121`; High is `Demand > 121`.
- **Chronological split:** 60,800 training records and 15,200 later test records; testing starts 2023-09-01.
- **Test accuracy:** 78.37%.
- **Balanced accuracy:** 78.34%.
- **Macro F1:** 0.7858.
- **Most-frequent-class baseline accuracy:** 36.25%.

The classifier performs better than the simple baseline on this held-out test period. Demand is not used as an input feature; the class is derived from demand solely as the target. The full classification report and confusion matrix are saved beside this file.

**Interpretation note:** this classifies recorded demand levels from the other supplied columns. Because `Units Sold` is strongly associated with `Demand`, verify the timing of sales information before treating it as a true future prediction.
