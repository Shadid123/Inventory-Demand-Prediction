# Clustering Findings

- **Algorithm:** K-Means.
- **Features:** Inventory Level, Units Sold, Units Ordered, Price, Discount, Competitor Pricing, Promotion, Epidemic.
- **Excluded:** `Demand` is not used to form clusters.
- **Number of clusters selected:** 2, selected from k=2..6 using the highest silhouette score on a deterministic sample from training-period rows.
- **Training selection silhouette:** 0.2581.
- **Held-out test-sample silhouette:** 0.2615 (computed on 3,000 later test records).

## Cluster profiles

The cluster number is an arbitrary group identifier, not a ranking. Compare the original-unit profile summaries:

- **Cluster 0:** 25,124 records (33.06%); mean units sold 103.57; mean discount 17.42; promotion active in 99.51% of records; mean units ordered 141.88.
- **Cluster 1:** 50,876 records (66.94%); mean units sold 81.55; mean discount 4.97; promotion active in 0.00% of records; mean units ordered 63.02.

For this run, cluster profiles mainly separate records with different promotion/discount and associated sales/order levels. This is a description of patterns in this dataset, not proof that promotion caused higher demand. See `cluster_profiles.csv`, `cluster_assignments.csv`, `kmeans_k_selection.csv`, and the figures for detail.
