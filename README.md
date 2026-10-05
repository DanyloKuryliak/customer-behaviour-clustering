# Customer Behaviour Clustering

Unsupervised-learning project: find useful credit-card customer profiles without a target column.

**Dataset:** [Credit Card Dataset for Clustering](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata), `CC GENERAL.csv`. It has 8,950 customers and 17 behavioural features covering balances, purchases, cash advances, payments, and tenure. `CUST_ID` is an identifier, so it is excluded from modelling.

## What the notebook does

1. Audits the raw data and missing values.
2. Median-imputes missing values, applies `log1p` to skewed amount/count columns, then standardises all features.
3. Tests K-Means, DBSCAN, and Gaussian Mixture Models (GMM).
4. Compares internal metrics and visualises results with PCA. PCA is for the 2D chart only; models use all 17 prepared features.

## Main result

**K-Means with K=4 is the recommended customer segmentation.** It is highly stable across random seeds (minimum ARI 0.993), has balanced cluster sizes (17.5%–30.8%), and creates four understandable profiles:

- occasional / low-activity customers;
- cash-advance-heavy, purchase-light customers;
- high-value, active purchasers;
- installment-focused purchasers.

K=2 makes the clearest broad split (silhouette 0.251), but K=4 is more useful for a business-facing segmentation. K=7 has a better Davies–Bouldin score but produces smaller, less simple groups.

## Model comparison

| Method | Best tested configuration | What we learned |
| --- | --- | --- |
| K-Means | `K=4`, K-Means++, `n_init=20` | Best choice for practical segmentation: stable, balanced, interpretable groups. |
| DBSCAN | `eps=2.120538`, `min_samples=20` | Found one broad dense region, a tiny 57-customer niche, and 5.82% noise. Better for density/outlier inspection than primary segmentation here. |
| GMM | 14 components, `full` covariance | Best probability-density fit by BIC/AIC, but 14 components include small groups and hard clusters overlap more. Useful for soft probabilities, not the clearest final segmentation. |

There is no ground-truth target or accuracy score. A good result here is stable, interpretable structure—not the highest single metric.

## Reproduce

The raw dataset is intentionally not versioned. Download it into `data/` using Kaggle, install dependencies with `uv pip install -r requirements.txt`, then run `customer-clustering.ipynb` from top to bottom.
