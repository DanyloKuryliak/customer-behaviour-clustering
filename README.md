# Customer Behaviour Clustering

Portfolio project: discover useful, stable credit-card customer profiles without
a target column or predefined customer types.

**Final recommendation:** use **K-Means with K=4** as the practical customer
segmentation. It produces balanced, stable, interpretable profiles while the
other methods provide useful alternative views of the same data.

**Dataset:** [Credit Card Dataset for Clustering](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata), `CC GENERAL.csv`. It has 8,950 customers and 17 behavioural features covering balances, purchases, cash advances, payments, and tenure. `CUST_ID` is an identifier, so it is excluded from modelling.

## Project workflow

1. Audits data quality, missing values, identifiers, and feature distributions.
2. Median-imputes missing numeric values, applies `log1p` to skewed amount and
   count features, then standardises all 17 behavioural features.
3. Tests K-Means, DBSCAN, and Gaussian Mixture Models (GMM) with controlled
   parameter searches.
4. Checks candidate K-Means segmentations across ten seeds using adjusted Rand
   index (ARI), profiles clusters in the original units, and uses PCA only for
   2D visual inspection.

## How results were judged

There is no ground-truth target and therefore no accuracy score. Selection is
based on complementary evidence:

| Evidence | What it measures | How it was used |
| --- | --- | --- |
| Silhouette score | How close members are to their own group compared with other groups | Higher is better for hard clustering. |
| Davies–Bouldin score | Cluster compactness relative to separation | Lower is better for hard clustering. |
| ARI across seeds | Whether a clustering changes when random initialisation changes | `1.0` means identical assignments. |
| BIC / AIC | GMM fit balanced against model complexity | Lower is better; not comparable to K-Means metrics. |
| Cluster size and profiles | Whether groups are large enough and understandable | Final practical decision criterion. |

## Results at a glance

**K-Means with K=4** is highly stable across random seeds (minimum ARI `0.993`),
has balanced cluster sizes (`17.5%`–`30.8%`), and creates four understandable
profiles:

- occasional / low-activity customers;
- cash-advance-heavy, purchase-light customers;
- high-value, active purchasers;
- installment-focused purchasers.

K=2 makes the clearest broad split (silhouette `0.251`), but K=4 gives more
actionable detail. K=7 has a better Davies–Bouldin score but produces smaller,
less simple groups.

## Model comparison

| Method | Best tested configuration | What we learned |
| --- | --- | --- |
| K-Means | `K=4`, K-Means++, `n_init=20` | Best choice for practical segmentation: stable, balanced, interpretable groups. |
| DBSCAN | `eps=2.120538`, `min_samples=20` | Found one broad dense region, a tiny 57-customer niche, and 5.82% noise. Better for density/outlier inspection than primary segmentation here. |
| GMM | 14 components, `full` covariance | Best probability-density fit by BIC/AIC, but 14 components include small groups and hard clusters overlap more. Useful for soft probabilities, not the clearest final segmentation. |

The methods are not competing on one shared score: K-Means and DBSCAN are hard
clustering methods, while GMM models probability distributions. A good result
here is stable, interpretable structure—not the highest single metric.

## Repository layout

```text
customer-clustering.ipynb  # executed analysis, evidence, and visualisations
README.md                  # project story and decisions
requirements.txt           # lightweight reproducibility dependencies
data/                      # local raw dataset; intentionally ignored by Git
```

## Reproduce

The raw dataset is intentionally not versioned. Download `CC GENERAL.csv` from
the dataset source into `data/`, then run:

```bash
uv venv
uv pip install --python .venv/bin/python -r requirements.txt
uv run --python .venv/bin/python jupyter nbconvert --to notebook --execute --inplace \
  customer-clustering.ipynb --ExecutePreprocessor.timeout=600
```

The saved notebook contains the complete audit, parameter-search evidence,
cluster profiles, PCA charts, and final comparison.

## Limitations

- Clusters are hypotheses, not verified customer labels.
- PCA charts are visual aids; every model is fitted on all 17 prepared features.
- Before a business action, validate that profiles remain stable on newer data
  and review them with a domain expert.
