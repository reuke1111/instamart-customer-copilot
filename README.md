# Overview

Part A of the Instamart Customer Intelligence & Support Copilot capstone: audit, clean, store and query deterministic synthetic data. All monetary values are in INR. This is fictional data and does not describe real Instamart operations. Part B adds measured customer segmentation and lapse prediction. Part C adds PII-safe offline ticket triage. Part D is not implemented.

The supplied scripts are preserved byte-for-byte from the assignment. `source_hashes.json` records their SHA-256 hashes; validation checks both. The generator also writes tickets, policy files and evaluation scenarios as required by its unmodified source; producing those fixtures does not implement the later Parts.

# Reproduce

Prerequisites: Python 3.11+ (the pinned dependencies were tested with Python 3.13.7), Git, and a shell at the repository root on branch `main`. Install dependencies once; all subsequent steps run offline. Run commands in this order (activation below is PowerShell):

1. `python -m venv .venv`
2. `.\.venv\Scripts\Activate.ps1`
3. `python -m pip install -r requirements.txt`
4. `python -m pip check`
5. `python generate_data.py`
6. `python -m scripts.audit_data`
7. `python -m scripts.clean_data`
8. `python -m scripts.sql_analysis`
9. `python -m scripts.exploratory_analysis`
10. `python -m scripts.build_readme`
11. `python scripts/validate_checkpoints.py`
12. `python -m scripts.run_part_b`
13. `python scripts/validate_part_b.py`
14. `python scripts/validate_checkpoints.py`
15. `python -m scripts.run_part_c`
16. `python scripts/validate_part_c.py`
17. `python scripts/validate_checkpoints.py`
18. `python scripts/validate_part_b.py`
19. `python -m unittest discover -s tests -v`

On macOS/Linux replace step 2 with `source .venv/bin/activate`. If PowerShell activation is unavailable, use `.venv\Scripts\python.exe` in place of `python` for steps 3–19. The local Windows Python alias was unavailable during setup, so `py -3.13 -m venv .venv` created the environment. `requirements.txt` was produced with `python -m pip freeze`; Direct dependencies are pandas, NumPy, scikit-learn and joblib; the frozen file also includes their required transitive dependencies.

Equivalent full regeneration after installation: `python -m scripts.run_part_a`, then `python -m scripts.run_part_b`, `python -m scripts.run_part_c` and `python -m unittest discover -s tests -v`. For an existing validated Parts A/B checkout, run only Part C; it validates both earlier Parts read-only and verifies that their artifacts are unchanged. The orchestrator verifies source hashes before running the generator. The validator reads artifacts without rewriting them and prints exactly `ALL CHECKPOINTS PASSED` on success. Do not use `python -O`, which disables assertions.

Measured runtime: Python 3.13.7, pandas 3.1.0, NumPy 2.3.5, SQLite 3.50.4.

# Part A

## Generation and raw audit

Generator output: `generated: customers=1200 products=40 orders_raw=13084 tickets=164`.

Missing values below use pandas' default CSV NA recognition: the literal `N/A` is missing, while `error` remains text until numeric coercion. Thus 53 raw amounts are already missing on load, but 120 amounts are non-numeric in total. Raw age, gender and duration blanks are counted separately. Tickets are not processed in Part A; any verification read uses `keep_default_na=False`.

**customers: 1,200 rows × 5 columns**

| Column | pandas dtype | Missing count | Missing % |
| --- | --- | --- | --- |
| customer_id | str | 0 | 0.0000 |
| city | str | 0 | 0.0000 |
| signup_date | str | 0 | 0.0000 |
| age | float64 | 61 | 5.0833 |
| gender | str | 30 | 2.5000 |

**products: 40 rows × 4 columns**

| Column | pandas dtype | Missing count | Missing % |
| --- | --- | --- | --- |
| product_id | str | 0 | 0.0000 |
| product_name | str | 0 | 0.0000 |
| category | str | 0 | 0.0000 |
| price_inr | int64 | 0 | 0.0000 |

**orders: 13,084 rows × 9 columns**

| Column | pandas dtype | Missing count | Missing % |
| --- | --- | --- | --- |
| order_id | str | 0 | 0.0000 |
| customer_id | str | 0 | 0.0000 |
| product_id | str | 0 | 0.0000 |
| quantity | int64 | 0 | 0.0000 |
| order_ts | str | 0 | 0.0000 |
| status | str | 0 | 0.0000 |
| discount_pct | int64 | 0 | 0.0000 |
| order_amount | str | 53 | 0.4051 |
| delivery_minutes | float64 | 2541 | 19.4207 |

| Audit metric | Measured value |
| --- | --- |
| raw orders | 13,084 |
| exact duplicates | 119 |
| pending rows | 656 |
| distinct order ids | 12,309 |
| non numeric amounts | 120 |
| negative delivery times | 106 |
| blank delivery times | 2,541 |
| blank ages | 61 |
| invalid ages | 6 |
| raw city spellings | 24 |
| nonblank gender spellings | 10 |
| blank genders | 30 |

## Cleaning rules and changes

Operations run in the order shown. Counts for parsing denote values converted to a new type; counts for replacement and imputation denote values changed; duplicate counts denote rows removed. Customer IDs are preserved exactly, and the product catalogue is unchanged.

| Customer operation | Values changed or parsed |
| --- | --- |
| Strip and title-case city | 476 |
| Map gender to male/female/unknown | 950 |
| Replace ages outside 10-100 with missing | 6 |
| Impute missing ages with city median | 67 |
| Parse signup_date to datetime | 1,200 |

Ages outside 10–100 are replaced before computing each normalized city's median with `groupby('city')['age'].transform('median')`. The 61 original blanks and 6 invalid ages make 67 imputations. No global median or rounded age is substituted.

| City | Customers | Age median used |
| --- | --- | --- |
| Bengaluru | 197 | 42.0 |
| Chennai | 191 | 39.0 |
| Delhi | 210 | 43.0 |
| Hyderabad | 191 | 37.5 |
| Mumbai | 184 | 38.5 |
| Pune | 227 | 39.0 |

| Gender | Customers |
| --- | --- |
| female | 610 |
| male | 560 |
| unknown | 30 |

Final ages: 0 missing; minimum 18, maximum 60. All 1,200 identifiers survive unchanged.

| Order operation | Rows/values changed or parsed |
| --- | --- |
| Parse order_ts to datetime | 13,084 |
| Remove exact duplicate rows | 119 |
| Keep latest snapshot per order ID | 656 |
| Convert order_amount to numeric | 12,258 |
| Recover amounts from price x quantity x (1-discount/100) | 112 |
| Replace negative delivery durations with missing | 105 |
| Impute completed durations with completed category median | 774 |
| Keep cancelled delivery durations missing | 0 |

After exact duplicates and earlier snapshots are removed, 12,258 nonmissing amount strings are converted: 61 become newly missing, while 51 were already missing (`N/A`). All 112 missing amounts are recovered with `round(price_inr * quantity * (1 - discount_pct / 100), 2)`.

Negative delivery times are made missing before category medians are calculated. Only completed orders participate in `groupby(category).transform('median')`. Of the 774 completed imputations, 669 were already blank and 105 were negative. Cancelled orders receive no imputation: 0 nonmissing values are cleared and 1,199 remain missing. Conflicting snapshots at the same timestamp, unknown products, pending final snapshots and groups without valid imputation observations raise errors.

| Product category | Completed delivery median (minutes) |
| --- | --- |
| Dairy | 25.0 |
| Electronics | 25.0 |
| Grocery | 25.0 |
| Personal Care | 25.5 |
| Snacks | 25.0 |

| Final checkpoint | Measured value |
| --- | --- |
| Orders | 12,309 |
| Completed | 11,110 |
| Cancelled | 1,199 |
| Pending | 0 |
| Missing amounts | 0 |
| Missing completed delivery times | 0 |
| Missing cancelled delivery times | 1,199 |
| Completed revenue (INR) | 2,998,668.80 |

## Storage and automated validation

`data/clean/customers.csv`, `products.csv` and `orders.csv` contain the clean tables. `data/instamart.db` contains `customers`, `products` and `orders`, with primary keys, foreign keys and domain checks. Both signup dates and order timestamps use `YYYY-MM-DD HH:MM:SS`; cancelled durations are SQL NULL. A complete temporary SQLite file replaces the previous database only after successful insertion. Re-running the stages replaces outputs rather than appending.

`scripts/validate_checkpoints.py` asserts all deterministic Part A reference counts, sums, rankings and EDA checkpoints, with explicit tolerances only for rounded or approximate statistics. It also checks raw-to-clean transformation, CSV/SQLite agreement, keys, NULLs, timestamps, all saved reports and this README. Standard-library unittest cases cover cleaning edge cases, SQL semantics, SQLite round-trips, independent clean-folder reproduction and detection of tampered outputs. Test failures are not suppressed.

## Seven SQL queries

All statements are in `sql/queries.sql`; each runs using `pd.read_sql_query`. Full result sets are saved as `outputs/part_a/q1.csv` through `q7.csv`. Revenue/spend includes only completed orders. Q3 uses the full catalogue.

| Query | Measured result |
| --- | --- |
| Q1 WHERE then HAVING | 30 customers: individual amounts > INR 500; their subtotal > INR 8,000 |
| Q2 Three-table join | 30 cells; INR 2,998,668.80; highest Bengaluru / Electronics, INR 233,909.45 |
| Q3 Scalar subquery | 11 products above average catalogue price |
| Q4 LEFT JOIN | 25 customers with no completed orders |
| Q5 Two CTEs | 550 customers above mean completed spending of INR 2,552.06 |
| Q6 Ranking windows | 1,175 customers with completed orders; 8 rows where RANK differs from ROW_NUMBER |
| Q7 Cumulative SUM | 183 days; final INR 2,998,668.80 |

Q5 averages the 1,175 customers with completed orders; customers with no completed orders are not added as zero spenders. Spending totals are rounded to cents before ranking so floating-point summation cannot split monetary ties. Ties share RANK (with subsequent gaps) and DENSE_RANK (without gaps), while ROW_NUMBER assigns a distinct position to every row, explaining the eight differences. Within a tied group ROW_NUMBER order is not a business distinction.

| City | Top customer | Completed spend (INR) |
| --- | --- | --- |
| Bengaluru | C0082 | 12,856.20 |
| Chennai | C0930 | 10,151.40 |
| Delhi | C0291 | 13,742.60 |
| Hyderabad | C0743 | 12,730.05 |
| Mumbai | C0693 | 14,766.55 |
| Pune | C0426 | 10,377.10 |

## pandas cross-check and EDA

The following `pivot_table(margins=True)` reproduces all 30 SQL Q2 cells (checked individually as well as in total). All values are INR.

| City | Dairy | Electronics | Grocery | Personal Care | Snacks | All |
| --- | --- | --- | --- | --- | --- | --- |
| Bengaluru | 87,075.05 | 233,909.45 | 110,915.65 | 20,193.85 | 71,549.40 | 523,643.40 |
| Chennai | 78,301.80 | 195,508.80 | 93,817.45 | 27,675.40 | 68,325.70 | 463,629.15 |
| Delhi | 117,271.70 | 158,549.10 | 112,658.05 | 23,471.50 | 90,879.45 | 502,829.80 |
| Hyderabad | 83,731.00 | 228,913.10 | 109,475.40 | 24,800.05 | 81,670.35 | 528,589.90 |
| Mumbai | 75,209.85 | 214,738.40 | 78,970.15 | 38,018.00 | 72,122.10 | 479,058.50 |
| Pune | 90,349.55 | 187,629.25 | 108,456.60 | 34,275.15 | 80,207.50 | 500,918.05 |
| All | 531,938.95 | 1,219,248.10 | 614,293.30 | 168,433.95 | 464,754.50 | 2,998,668.80 |

SQL grand total = pandas grand total = INR 2,998,668.80.

Named aggregation uses `revenue_sum=('order_amount', 'sum')`, `revenue_mean=('order_amount', 'mean')` and `completed_order_count=('order_amount', 'count')`.

| City | Revenue sum (INR) | Mean amount (INR) | Completed count |
| --- | --- | --- | --- |
| Bengaluru | 523,643.40 | 276.47 | 1894 |
| Chennai | 463,629.15 | 272.72 | 1700 |
| Delhi | 502,829.80 | 243.62 | 2064 |
| Hyderabad | 528,589.90 | 284.95 | 1855 |
| Mumbai | 479,058.50 | 297.00 | 1613 |
| Pune | 500,918.05 | 252.48 | 1984 |

| Completed amount statistic | Measured value |
| --- | --- |
| Mean (INR) | 269.91 |
| Median / Q2 (INR) | 135.00 |
| Skewness (pandas unbiased sample skew) | 4.608827 |
| Q1 (INR) | 64.80 |
| Q3 (INR) | 297.00 |
| IQR (INR) | 232.20 |
| Lower 1.5 x IQR fence (INR) | -283.50 |
| Upper 1.5 x IQR fence (INR) | 645.30 |
| Below lower fence | 0 |
| Above upper fence | 897 |

Pearson correlation matrix, over all 11,110 completed orders after cleaning:

| Variable | order_amount | discount_pct | quantity | delivery_minutes |
| --- | --- | --- | --- | --- |
| order_amount | 1.000000 | -0.095106 | 0.087294 | -0.003135 |
| discount_pct | -0.095106 | 1.000000 | -0.000359 | -0.009808 |
| quantity | 0.087294 | -0.000359 | 1.000000 | -0.012623 |
| delivery_minutes | -0.003135 | -0.009808 | -0.012623 | 1.000000 |

**Four EDA answers (under 200 words total)**

**What is this data about?** Synthetic Instamart orders across six cities and five product categories from April to September 2025. Revenue and delivery analysis use completed orders only.

**What do its distributions and statistics look like?** Amounts are right-skewed: mean INR 269.91, median INR 135.00, and skewness 4.61. The middle half lies between INR 64.80 and INR 297.00.

**Are there outliers or anomalies?** 897 completed amounts exceed the upper IQR fence of INR 645.30; 0 fall below the lower fence. High-priced electronics can explain large valid purchases, so outliers are retained. Raw duplicates, earlier pending snapshots, invalid ages, amounts and durations are corrected using the documented rules.

**What relationships exist between variables?** The largest absolute Pearson correlation is order_amount with discount_pct (-0.095106), a weak negative association. The amount formula includes discount, while product prices and quantities also vary. Correlation of this size does not establish causation.

## Artifact map

- Supplied sources: `generate_data.py`, `mock_llm.py`, `source_hashes.json`.
- Modular stages: `scripts/audit_data.py`, `clean_data.py`, `sql_analysis.py`, `exploratory_analysis.py`, `build_readme.py`, `validate_checkpoints.py`, `run_part_a.py`.
- Shared helpers and fixed references: `scripts/common.py`, `scripts/checkpoints.py`.
- Audit/cleaning/SQL/EDA JSON reports, seven query CSVs, pivot, city aggregates and correlation matrix: `outputs/part_a/`.
- Raw and clean CSVs: `data/raw/`, `data/clean/`; SQLite: `data/instamart.db`.
- Tests: `tests/`; dependencies: `requirements.txt`; Git exclusions: `.gitignore`; source-byte preservation: `.gitattributes`.

# Part B

Completed: customer segmentation and September lapse prediction, using the verified Part A clean tables. No external API or paid service is used. Model runtime: scikit-learn 1.7.2, joblib 1.5.2.

## RFM and K-means

Only completed orders before the snapshot `2025-10-01 00:00:00` contribute to RFM. Recency uses whole elapsed days, frequency counts distinct orders, and monetary sums completed order amounts. Customer IDs are sorted before all clustering and split operations.

| RFM checkpoint | Measured |
| --- | --- |
| Customers | 1175 |
| Recency median / minimum / maximum (days) | 19 / 0 / 180 |
| Frequency mean / maximum | 9.455319 / 33 |
| Monetary mean / maximum (INR) | 2,552.06 / 14,766.55 |

`StandardScaler` is fitted to the three RFM columns for this descriptive whole-window segmentation. Every candidate uses `KMeans(n_clusters=k, n_init=10, random_state=42)`.

| K | Measured inertia | Measured silhouette |
| --- | --- | --- |
| 2 | 1820.7606 | 0.464972 |
| 3 | 1268.2346 | 0.445161 |
| 4 | 743.0711 | 0.525544 |
| 5 | 617.3666 | 0.480495 |
| 6 | 503.1706 | 0.481872 |
| 7 | 408.4605 | 0.416337 |
| 8 | 366.5035 | 0.412329 |

Chosen K: **4**, the highest measured silhouette among the permitted K values 3, 4 and 5. For these data, the large inertia drop from K=3 to K=4 is followed by a smaller gain at K=5, and four profiles support distinct retention actions. Adding clusters allows smaller within-cluster distances, explaining the lower measured inertia at K=8 than K=2; inertia alone does not penalise complexity or establish useful business segments. K-means finds local solutions, so monotonicity is a theoretical optimum property rather than a guarantee for every individual fit.

| ID | Name | Customers | Mean recency | Mean frequency | Mean monetary (INR) | Action |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | Occasional | 436 | 28.86 | 3.77 | 1,057.62 | Encourage a second shopping occasion with relevant basket suggestions. |
| 1 | Premium | 138 | 24.12 | 6.77 | 7,129.70 | Offer priority service and personalised high-value retention outreach. |
| 2 | Frequent | 419 | 8.77 | 19.40 | 3,410.02 | Reward repeat grocery shopping with replenishment reminders and loyalty benefits. |
| 3 | Dormant | 182 | 109.04 | 2.20 | 685.99 | Test a limited win-back offer and ask what stopped repeat purchases. |

Cluster numbers are arbitrary labels with no ordering. Names are assigned from measured profiles: Premium has the highest mean monetary value; Frequent has the highest remaining frequency; Dormant has the highest remaining recency. Only the highest-mean-monetary cluster has `is_premium=True`.

`PCA(n_components=2)` on scaled RFM explains 0.634279 (PC1) and 0.218038 (PC2), for 0.852316 combined. PCA is descriptive and is not a lapse-model feature.

## Lapse cohort, features and leakage controls

Eligibility requires at least one completed order strictly before `2025-09-01 00:00:00`. The target is 1 for no completed order in `[2025-09-01, 2025-10-01)`, otherwise 0. September cancellations do not count as activity. No September orders enter feature aggregation.

| Cohort checkpoint | Measured |
| --- | --- |
| Eligible customers | 1156 |
| Lapsed customers | 443 |
| Lapse rate | 38.32% |
| Training / test rows | 924 / 232 |
| Training non-lapsed / lapsed | 570 / 354 |
| Test non-lapsed / lapsed | 143 / 89 |
| Missing feature values | 0 |

Numeric features: `recency`, `frequency`, `monetary`, `average_discount`, `cancellation_rate`, `average_delivery`, `age`, `tenure`.
Categorical features: `favourite_category`, `city`, `gender`.

RFM, average discount and average delivery use pre-cutoff completed orders. Cancellation rate is cancelled divided by all final-status pre-cutoff orders. Favourite category is the most frequent completed category, with alphabetical tie-breaking. Tenure is whole days from signup to cutoff. City, age and gender come from the required clean customer table. Features are calculated from the fixed Part A clean records; Part A's historical cleaning is not rerun.

The split is `train_test_split(test_size=0.2, stratify=y, random_state=42)`. A `ColumnTransformer` inside each model `Pipeline` contains numeric `SimpleImputer(strategy='median')` then `StandardScaler`, and categorical `SimpleImputer(strategy='most_frequent')` then `OneHotEncoder(drop='first', handle_unknown='ignore')`. The supplied features have no missing values; the imputers also handle future missing inputs using training-fitted statistics. Every encoder, scaler and model is fitted within each CV training fold and then refitted on the full training split only. No encoder, scaler or resampler sees test rows during fitting; the test set is only transformed and evaluated. No target-derived features, whole-window Premium flags, or segment labels enter the feature schema.

The fitted frequency scaler mean is 8.114718615; the full-cohort mean is 8.028546713. They differ because the scaler sees only the 924 training customers, not all 1,156 cohort customers.

Transformed model features: `numeric__recency`, `numeric__frequency`, `numeric__monetary`, `numeric__average_discount`, `numeric__cancellation_rate`, `numeric__average_delivery`, `numeric__age`, `numeric__tenure`, `categorical__favourite_category_Electronics`, `categorical__favourite_category_Grocery`, `categorical__favourite_category_Personal Care`, `categorical__favourite_category_Snacks`, `categorical__city_Chennai`, `categorical__city_Delhi`, `categorical__city_Hyderabad`, `categorical__city_Mumbai`, `categorical__city_Pune`, `categorical__gender_male`, `categorical__gender_unknown`.

## Baseline, tuning and held-out evaluation

The baseline always predicts the most frequent training class (not lapsed). Both grid searches use `GridSearchCV(cv=5, scoring='f1')`, with positive class `lapsed=1`. Logistic regression uses `max_iter=1000` and `model__C=[0.01, 0.1, 1, 10, 100]` (25 CV fits). The forest uses `n_estimators=100, random_state=42` and `model__max_depth=[3,5,8]` × `model__min_samples_leaf=[5,10,20]` (45 CV fits). Full CV result tables are saved. Both searches finish before test evaluation, and model-family selection uses the higher best training CV F1.

| Model | Best parameters | CV F1 | Test accuracy | Test precision | Test recall | Test F1 |
| --- | --- | --- | --- | --- | --- | --- |
| Majority baseline | Most frequent class | — | 0.6164 | 0.0000 | 0.0000 | 0.0000 |
| logistic | {'model__C': 100} | 0.6506 | 0.7284 | 0.6444 | 0.6517 | 0.6480 |
| forest | {'model__max_depth': 8, 'model__min_samples_leaf': 10} | 0.6865 | 0.7241 | 0.6404 | 0.6404 | 0.6404 |

Confusion matrices use actual rows [not lapsed, lapsed] and predicted columns [not lapsed, lapsed], i.e. `[[TN, FP], [FN, TP]]`.

| Model | Confusion matrix |
| --- | --- |
| Majority baseline | [[143, 0], [89, 0]] |
| logistic | [[111, 32], [31, 58]] |
| forest | [[111, 32], [32, 57]] |

The selected model is **forest**, chosen for its higher training CV F1 (0.6865) before examining holdout performance. Its test F1 of 0.6404 exceeds the required 0.55. Missing a lapsing customer loses a retention opportunity, while targeting a loyal customer spends incentives unnecessarily. The logistic model's slightly different holdout score does not justify retuning or switching after seeing the test set; campaign costs and future validation should determine any threshold change.

Reference discrepancy: the measured logistic optimum is `C=100`, compared with the brief's illustrative value around 1. All prescribed candidates were evaluated on the fixed split; results are reported as measured and no hyperparameters were changed to match the illustrative reference.

## Training-only oversampling

All 924 original training rows are retained; 216 lapsed training rows are sampled with replacement using `random_state=42`. This yields 570 non-lapsed and 570 lapsed rows. The best logistic configuration is cloned and refitted on that balanced training set, then evaluated on the same untouched 232 test rows.

| Logistic experiment | Precision | Recall | F1 | Confusion matrix |
| --- | --- | --- | --- | --- |
| Original | 0.6444 | 0.6517 | 0.6480 | [[111, 32], [31, 58]] |
| Oversampled | 0.5891 | 0.8539 | 0.6972 | [[90, 53], [13, 76]] |

The largest absolute change is recall: recall changes by +0.2022, precision by -0.0553, and F1 by +0.0492. More lapsing customers are found, at the cost of more loyal customers receiving unnecessary outreach. This holdout experiment does not retroactively change the CV-selected persisted model.

## Interpretation, persistence and retention

| Forest feature | Importance | Business reading |
| --- | --- | --- |
| numeric__monetary | 0.277830 | Historical spend helps distinguish purchase patterns; importance has no direction. |
| numeric__frequency | 0.276787 | Purchase regularity is informative about lapse risk. |
| numeric__recency | 0.111219 | Time since the last purchase helps identify disengagement. |
| numeric__average_delivery | 0.061649 | Delivery experience is associated with behaviour; this is not a causal effect. |
| numeric__average_discount | 0.053178 | Discount usage helps distinguish customer shopping patterns. |

Impurity-based feature importances are associations and can favour continuous variables; they are not causal effects.

| Logistic feature | Coefficient | Sign | Business reading |
| --- | --- | --- | --- |
| numeric__recency | 0.039585 | positive | Higher recency is associated with higher lapse log-odds, holding other inputs fixed. |
| numeric__frequency | -1.698239 | negative | Higher frequency is associated with lower lapse log-odds, holding other inputs fixed. |
| numeric__monetary | 0.284573 | positive | Higher monetary is associated with higher lapse log-odds, holding other inputs fixed. |

Numeric coefficients correspond to a one-training-standard-deviation change. RFM variables are correlated, so these conditional signs should not be read as independent causal effects.

`outputs/lapse_model.joblib` stores the selected full preprocessing-and-model pipeline. Reloading it reproduces all 232 test predictions and probabilities exactly; both equality checks passed. Only trusted locally generated joblib artifacts are loaded by this project.

There are 40 Premium test customers; the top 10 by saved-model lapse probability are retained, and 1 actually lapsed. Tied probabilities are ordered by customer ID.

| Customer ID | Lapse probability | Actually lapsed |
| --- | --- | --- |
| C0862 | 0.530369 | 0 |
| C0060 | 0.476473 | 0 |
| C1008 | 0.475621 | 0 |
| C0267 | 0.475263 | 0 |
| C0133 | 0.471179 | 0 |
| C0942 | 0.468798 | 0 |
| C1000 | 0.465437 | 0 |
| C0553 | 0.464462 | 0 |
| C1112 | 0.462014 | 0 |
| C0565 | 0.460255 | 1 |

Premium segments use the October snapshot, including September outcomes: the Premium filter contains future information, so this retention list is an illustrative post-hoc analysis, not a leakage-free prospective evaluation. Premium status and segment labels are never predictive features.

## Part B artifacts and verification

- Required outputs: `outputs/customer_segments.csv`, `outputs/lapse_model.joblib`, `outputs/retention_list.csv`.
- Measured reports, RFM profiles, K comparison, PCA coordinates, cohort, split assignments, CV scores, holdout predictions, interpretation and Part A preservation hashes: `outputs/part_b/`.
- Stages: `scripts/segmentation.py`, `scripts/lapse_features.py`, `scripts/lapse_models.py`, `scripts/run_part_b.py`.
- Checks: `scripts/validate_part_b.py` and `tests/test_part_b.py`. Part A numerical assertions remain intact; tests include full A-to-B reproduction in an isolated temporary folder.

# Part C

Completed: offline support ticket triage with input guardrails, PII redaction, retries, response validation, run-wide token budgets and Part B-aware routing. The supplied `mock_llm.py` is unchanged; no external model, account or API is used.

## Inputs, guardrails and redaction

`data/raw/tickets.csv` is read with `keep_default_na=False` and processed in its original row order. Empty or whitespace-only tickets are skipped; tickets exceeding `MAX_INPUT_TOKENS=400` using `mock_llm.est_tokens` are flagged before any model call. Guard decisions use original input length, so redaction cannot make a long ticket bypass the limit.

| Checkpoint | Measured |
| --- | --- |
| Total tickets | 164 |
| Empty / skipped | 4 |
| Long / flagged | 4 |
| Tokens per long ticket | [734, 734, 734, 734] |
| Eligible tickets | 156 |
| Email replacements | 55 |
| Phone replacements | 91 |
| Order ID replacements | 132 |
| Total replacements | 278 |
| Remaining PII pattern matches | 0 |

Redaction applies the assignment's patterns in exactly this order: `[\w.+-]+@[\w-]+\.[\w.]+` → `[EMAIL]`; `\b[6-9]\d{9}\b` → `[PHONE]`; `ORD-\d{6}` → `[ORDER_ID]`. Counts cover all 164 input tickets once, including guarded tickets. Reapplying the patterns finds zero matches. Every user message equals the redacted ticket text alone; ground-truth labels, customer segments and identifiers are never added to prompts.

## Prompts, retries and response validation

Both system prompts contain CLASSIFY, define billing/delivery/technical/fraud and request JSON-only category/reason fields. The zero-shot prompt has no EXAMPLES: marker. The few-shot prompt includes labelled fraud and technical examples. Exact prompts are saved under `outputs/part_c/`, and the mock is reset immediately before each independent run.

Each ticket permits an initial attempt plus at most three retries for `TransientLLMError`. Retry n uses `0.05 * 2**n * uniform(0.9, 1.15)` for n=0,1,2; a per-run random seed of 42 makes the jitter reproducible. One JSON log line per retry records only ticket ID, retry number and wait. Exception messages, ticket bodies and model replies are never persisted.

| Run | Model attempts | Transient failures | Retries | Permanent failures |
| --- | --- | --- | --- | --- |
| zero_shot | 175 | 19 | 19 | 0 |
| few_shot | 175 | 19 | 19 | 0 |
| budget_demo | 97 | 10 | 10 | 0 |
| final | 175 | 19 | 19 | 0 |

For each full run, all 19 transient failures recover on retry 0; every wait is between 0.045 and 0.0575 seconds. Exhausting retries would flag the ticket for human review; this does not occur with the supplied dataset.

Valid JSON is accepted only when its category is one of the four allowed strings. Non-JSON replies use `re.findall(r'[a-z]+', text.lower())` and are accepted only when exactly one distinct allowed category occurs. Thus hyphenated category words are recognised, while valid JSON with an invalid category never gets keyword salvage.

| Full-run parsing outcome | Count |
| --- | --- |
| classified_json | 135 |
| classified_keyword | 12 |
| flagged_invalid_output | 9 |
| Successfully classified | 147 |

## Accuracy and token comparison

| Prompt | Correct / classified | Accuracy | Total tokens | Tokens / classified | System prompt tokens |
| --- | --- | --- | --- | --- | --- |
| Zero-shot | 126/147 | 85.71% | 10804 | 73.4966 | 41 |
| Few-shot | 147/147 | 100.00% | 17044 | 115.9456 | 81 |

Zero-shot confusion matrix: rows are actual categories, columns are predictions. Only the 147 classified tickets enter this matrix and the accuracy denominator.

| Actual / predicted | billing | delivery | technical | fraud |
| --- | --- | --- | --- | --- |
| billing | 38 | 0 | 0 | 0 |
| delivery | 0 | 36 | 0 | 0 |
| technical | 10 | 0 | 29 | 0 |
| fraud | 11 | 0 | 0 | 23 |

The few-shot prompt raises measured accuracy from 85.71% to 100.00%, correcting 11 fraud-to-billing and 10 technical-to-billing errors. It consumes 6,240 extra estimated tokens (57.76% more), increasing tokens per classified ticket from 73.50 to 115.95. That increase is worthwhile when missing fraud or sending app failures to billing costs more than the added processing. These are offline token estimates, not monetary charges or evidence of real-model accuracy.

## Budget accounting

Sum actual response usage across every attempt, including successful retries and invalid outputs. The supplied mock's transient exceptions carry no usage, so they add zero reported tokens. Check the cumulative total before each call; the last response can cross the threshold.

The ledger charges successful retries and invalid replies exactly once. The budget is shared across all tickets and checked again before every retry; empty and long tickets retain their input-guard status even after exhaustion. Failures without returned usage are counted as failures, not assigned fabricated token estimates.

| Budget demonstration | Measured |
| --- | --- |
| Configured budget | 6000 |
| Actual reported usage | 6033 |
| Budget-exhausted tickets | 69 |
| Classified tickets | 82 |
| Invalid replies | 5 |
| Retry attempts | 10 |

The last permitted response takes usage to 6,033, which is 33 above the 6,000 threshold; no subsequent model call is made. This implements the specified pre-call threshold check, rather than a reservation of unknown future response tokens.

## Final routing and artifacts

The final independent few-shot run uses a 100,000-token budget, consumes 17,044 tokens, and produces 164 rows. Guarded, invalid, exhausted or permanently failed tickets go to human_review/NORMAL. Otherwise fraud takes precedence over Premium and goes to security_team/HIGH; other Premium tickets go to premium_desk/HIGH; remaining tickets go to support_queue/NORMAL. Unknown customers are not Premium; unavailable category and segment names remain blank.

| Final route | Tickets |
| --- | --- |
| human_review | 17 |
| premium_desk | 11 |
| security_team | 34 |
| support_queue | 102 |

| Priority | Tickets |
| --- | --- |
| HIGH | 45 |
| NORMAL | 119 |

The 135 JSON and 12 keyword classifications passed rule-based output checks; this is not human validation. Human review receives 9 invalid replies, 4 empty tickets and 4 long tickets. Premium membership is read from the unchanged Part B segment file.

- Required outputs: `outputs/triage_results.csv` and `outputs/triage_budget_demo.csv`.
- Supporting outputs: `outputs/part_c/triage_summary.json`, zero/few-shot results, zero-shot confusion CSV, exact system prompts, per-attempt token ledgers, retry JSONL logs and Parts A/B preservation hashes.
- Implementation: `scripts/triage.py`, `scripts/triage_analysis.py`, `scripts/run_part_c.py`, `scripts/validate_part_c.py`, `scripts/part_c_integrity.py` and `scripts/part_c_report.py`.
- Tests: `tests/test_part_c.py` plus an extended fresh-folder A-through-C reproduction test. All earlier numerical assertions are retained.

## Privacy safeguards and limitations

The new Part C outputs contain no ticket bodies, model reply bodies, original email addresses, phone numbers or order IDs. Only the required synthetic ticket and customer identifiers remain for routing; these are not full anonymisation. Regex assertions scan every prompt and all new Part C artifacts. The fixed patterns cover this assignment's planted formats, not arbitrary real-world PII. The classifier is a deterministic keyword mock, so its few-shot gains do not establish production accuracy. The Part B Premium flag uses the whole-window segment and is illustrative for this synthetic routing exercise.

# Part D

Not yet implemented. Requires separate approval after Part C.

# Limitations and Next Steps

All data is synthetic and covers a fixed six-month window. City/category median imputation reduces observed variation; imputed delivery times are analytical estimates, not observed delivery evidence. Cancelled orders remain outside revenue and delivery analyses. Large amounts are retained because an IQR flag alone does not establish an error. Correlations are weak and do not establish causation.

SQLite REAL and pandas floats use binary arithmetic; comparisons use a tight absolute tolerance and reported currency/ranked spending is rounded to cents. For production financial accounting, use integer paise or decimal arithmetic. Raw-string dtype labels and formatting can vary across pandas versions, so the measured environment is pinned and README results are generated from outputs.

Premium segments use the October snapshot, including September outcomes: the Premium filter contains future information, so this retention list is an illustrative post-hoc analysis, not a leakage-free prospective evaluation. Premium status and segment labels are never predictive features. The model evaluation uses the fixed clean Part A inputs; Part A's already-completed imputations are inherited, rather than refitted by Part B. A production prospective study should also fit upstream cleaning on historical training information. Parts A, B and C are complete. Part C regex redaction covers the specified synthetic PII formats; the mock and token estimator are not production evaluations. Review Part C before approving Part D. A policy assistant and refund agent are not implemented. Publishing a public GitHub repository is not performed by this local pipeline.
