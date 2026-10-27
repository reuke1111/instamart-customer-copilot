# Overview

Part A of the Instamart Customer Intelligence & Support Copilot capstone: audit, clean, store and query deterministic synthetic data. All monetary values are in INR. This is fictional data and does not describe real Instamart operations. Part B adds measured customer segmentation and lapse prediction. Part C adds PII-safe offline ticket triage. Part D adds policy RAG and a resumable refund agent.

The supplied scripts are preserved byte-for-byte from the assignment. `source_hashes.json` records their SHA-256 hashes; validation checks both. The generator also writes tickets, policy files and evaluation scenarios as required by its unmodified source; producing those fixtures does not implement the later Parts.

# Reproduce

Prerequisites: Python 3.11 or 3.13, Git, and a shell at the repository root on branch `main`. Install dependencies once; all subsequent steps run offline. Run commands in this order (activation below is PowerShell):

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
19. `python -m scripts.run_part_d`
20. `python scripts/validate_part_d.py`
21. `python scripts/validate_checkpoints.py`
22. `python scripts/validate_part_b.py`
23. `python scripts/validate_part_c.py`
24. `python -m unittest discover -s tests -v`
25. `python -m scripts.independent_audit`

On macOS/Linux replace step 2 with `source .venv/bin/activate`. If PowerShell activation is unavailable, use `.venv\Scripts\python.exe` in place of `python` for steps 3–25. The local Windows Python alias was unavailable during setup, so `py -3.13 -m venv .venv` created the environment. `requirements.txt` pins the tested runtime; Direct dependencies are pandas, NumPy, scikit-learn and joblib; the frozen file also includes their required transitive dependencies.

Equivalent full regeneration after installation: `python -m scripts.run_part_a`, then `python -m scripts.run_part_b`, `python -m scripts.run_part_c`, `python -m scripts.run_part_d` and `python -m unittest discover -s tests -v`. For an existing validated Parts A/B/C checkout, run only Part D; it validates all earlier Parts read-only and verifies that their artifacts are unchanged. The orchestrator verifies source hashes before running the generator. The validator reads artifacts without rewriting them and prints exactly `ALL CHECKPOINTS PASSED` on success. Do not use `python -O`, which disables assertions.

Pinned runtime: pandas 2.3.3, NumPy 2.3.5, scikit-learn 1.8.0, joblib 1.5.2. Use `python -m scripts.runtime` to verify the active interpreter's packages. Always use the activated project environment, or its explicit Python path; a system `python` can load different packages. Existing environments must run `python -m pip install -r requirements.txt` before validation. Model files are version-specific: do not copy estimator internals or suppress version warnings to load a model with another scikit-learn version.

# Part A

## Generation and raw audit

Generator output: `generated: customers=1200 products=40 orders_raw=13084 tickets=164`.

Missing values below use pandas' default CSV NA recognition: the literal `N/A` is missing, while `error` remains text until numeric coercion. Thus 53 raw amounts are already missing on load, but 120 amounts are non-numeric in total. Raw age, gender and duration blanks are counted separately. Tickets are not processed in Part A; any verification read uses `keep_default_na=False`. Table CSVs use explicit object text, float64 and int64 schemas, so the audit reports actual dtypes consistently across pandas string-inference defaults. Numeric CSV input uses round-trip float parsing.

**customers: 1,200 rows × 5 columns**

| Column | pandas dtype | Missing count | Missing % |
| --- | --- | --- | --- |
| customer_id | object | 0 | 0.0000 |
| city | object | 0 | 0.0000 |
| signup_date | object | 0 | 0.0000 |
| age | float64 | 61 | 5.0833 |
| gender | object | 30 | 2.5000 |

**products: 40 rows × 4 columns**

| Column | pandas dtype | Missing count | Missing % |
| --- | --- | --- | --- |
| product_id | object | 0 | 0.0000 |
| product_name | object | 0 | 0.0000 |
| category | object | 0 | 0.0000 |
| price_inr | int64 | 0 | 0.0000 |

**orders: 13,084 rows × 9 columns**

| Column | pandas dtype | Missing count | Missing % |
| --- | --- | --- | --- |
| order_id | object | 0 | 0.0000 |
| customer_id | object | 0 | 0.0000 |
| product_id | object | 0 | 0.0000 |
| quantity | int64 | 0 | 0.0000 |
| order_ts | object | 0 | 0.0000 |
| status | object | 0 | 0.0000 |
| discount_pct | int64 | 0 | 0.0000 |
| order_amount | object | 53 | 0.4051 |
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

`data/clean/customers.csv`, `products.csv` and `orders.csv` contain the clean tables. `data/instamart.db` contains `customers`, `products` and `orders`, with primary keys, foreign keys and domain checks. Both signup dates and order timestamps use `YYYY-MM-DD HH:MM:SS`; cancelled durations are SQL NULL. A complete temporary SQLite file replaces the previous database only after successful insertion. If its full schema and rows equal the existing database, the original file is retained so SQLite writer-version header bytes cannot invalidate agent checkpoints on a no-op rebuild. Changed data or schema still replaces the database and invalidates old checkpoints. Re-running the stages replaces outputs rather than appending.

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

Completed: customer segmentation and September lapse prediction, using the verified Part A clean tables. No external API or paid service is used. Model runtime pins: scikit-learn 1.8.0, joblib 1.5.2. `outputs/part_b/model_metadata.json` records the training dependency versions and model SHA-256; the loader checks both before deserialization.

## RFM and K-means

Only completed orders before the snapshot `2025-10-01 00:00:00` contribute to RFM. Recency uses whole elapsed days, frequency counts distinct orders, and monetary sums completed order amounts in integer paise, converted to INR only after summation and written with two decimal places. Customer IDs are sorted before all clustering and split operations.

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

Completed: local policy retrieval and a bounded ReAct refund agent using the unchanged `mock_llm.py`, SQLite, scikit-learn and standard-library persistence. No external API or real payment service is used.

## Policy corpus, chunks and retrieval

The five synthetic policies cover refunds, delivery, escalation, payment and privacy. Whitespace splitting uses 40-word chunks with 10-word overlap (step 30), stopping as soon as a chunk reaches the document end. Files sort by filename; chunk IDs are `filename:000`, `filename:001`, etc. Word offsets are zero-based and end-exclusive.

| Source | Overlap 10 | Overlap 0 |
| --- | --- | --- |
| delivery_policy.txt | 3 | 3 |
| escalation_policy.txt | 3 | 2 |
| payment_policy.txt | 3 | 3 |
| privacy_policy.txt | 2 | 2 |
| refund_policy.txt | 4 | 3 |
| Total | 15 | 13 |

`TfidfVectorizer()` is fitted to chunk text only. Cosine similarity ranks chunks; ties sort by chunk ID. `retrieve(query, k=3)` returns source, score and text. There are no query-to-document mappings. For `refund policy late delivery`:

| Rank | Source | Cosine score | Opening words |
| --- | --- | --- | --- |
| 1 | refund_policy.txt | 0.271928151 | order is delivered in more than 45 minutes, the customer receives a |
| 2 | refund_policy.txt | 0.251374134 | Refund Policy. A refund can be requested only for orders whose status |
| 3 | refund_policy.txt | 0.245405809 | late-delivery refund is paid. Refunds up to INR 1000 are approved automatically |

TF measures how often a term occurs within a chunk; IDF gives more weight to terms that appear in fewer chunks. TF-IDF depends on shared vocabulary and can miss a paraphrase with the same meaning but different words.

`answer(question)` sends exactly three filename-labelled chunks under `CONTEXT:`, then the original question under `QUESTION:`; its system prompt contains POLICY ASSISTANT. Only the supplied mock answers. Extracted words are traced to their original source positions: proven duplicate overlap is removed, while cross-document joins and gaps are rejected with an unsupported-information fallback. No replacement facts are selected from unretrieved text. Raw model text remains in the evaluation evidence. Evaluation references are used only for scoring, never supplied to the model.

## Measured RAG evaluation

| Overlap | Questions | Source retrieval | Raw mock facts | Validated answer facts | Unsupported fallbacks | Tokens |
| --- | --- | --- | --- | --- | --- | --- |
| 10 | 14 | 12/12 | 12/12 | 12/12 | 2/2 | 3234 |
| 0 | 14 | 12/12 | 10/12 | 8/12 | 2/2 | 3108 |

Without overlap, the unchanged raw mock reproduces 10/12 expected facts, but source-boundary validation accepts only 8/12. The original 10/12 substring score counted misleading mixed-source fragments as successes. This justified safety correction is not a retrieval regression: source coverage remains 12/12, and the original rubric requires at least one zero-overlap failure. Measured validated failures:

| Question | Missing expected fact | Actual answer |
| --- | --- | --- |
| What delivery time makes a completed order eligible for a full refund? | more than 45 minutes | receives a full refund of the order amount. |
| What is the delivery fee on small orders? | INR 25 | I could not find that information in the policy documents. |
| How soon is money returned when a payment fails but is deducted? | 48 hours | I could not find that information in the policy documents. |
| When are delivery addresses deleted from active systems? | 180 days | I could not find that information in the policy documents. |

In refund_policy.txt, the 40-word boundary separates the late-delivery condition from its refund consequence. In payment_policy.txt it separates 'returned automatically' from 'within 48 hours'. Retrieval still finds the right documents, but isolated sentence fragments lose the required facts; concatenation can even join unrelated fragments from different documents; the new check rejects those joins, including the delivery-fee and address-deletion fragments followed by unrelated policy titles. The ten repeated words restore complete facts for this corpus. The two unsupported questions fall back in both configurations.

## Tools and execution rules

| Tool | Parameters | Behavior |
| --- | --- | --- |
| get_order | order_id | Read-only, parameterized SQLite lookup; four fields or a structured missing-order error. |
| search_policy | query | Reuse TF-IDF retrieval; return three chunks. |
| calculate_refund | order_amount, delivery_minutes | Verify inputs against the authorized order; return full amount only after 45 minutes. |
| issue_refund | order_id, refund_inr | Record eligible amount up to and including INR 1,000. |
| escalate_to_human | order_id, refund_inr | Record eligible amount above INR 1,000 for approval; do not issue it. |

`TOOLS_SPEC` includes descriptions and object schemas with required fields and no additional properties; `TOOL_MAP` is an explicit allowlist. Runtime validation rejects unknown tools, malformed arguments, booleans masquerading as numbers, nonfinite or negative values, fractional paise and order IDs outside the original request. Each financial execution independently re-reads the completed order, its delivery duration and exact amount. Decimal checks enforce the amount and approval boundary.

Missing orders yield NOT_FOUND; cancelled orders and completed deliveries at or below 45 minutes are denied. Completed deliveries strictly above 45 minutes receive the full amount through the appropriate route. As instructed, all supplied scenarios are considered within the seven-day policy window; date eligibility is not inferred from today's date or enforced for arbitrary production requests.

## Seven normal scenarios

| Scenario | Status | Refund / escalation INR | LLM calls | Tokens | Tool sequence |
| --- | --- | --- | --- | --- | --- |
| S1 | REFUND | 128.25 | 5 | 894 | get_order → search_policy → calculate_refund → issue_refund |
| S2 | REFUND | 135.00 | 5 | 894 | get_order → search_policy → calculate_refund → issue_refund |
| S3 | ESCALATE | 1999.00 | 5 | 881 | get_order → search_policy → calculate_refund → escalate_to_human |
| S4 | DENIED | 0.00 | 4 | 615 | get_order → search_policy → calculate_refund |
| S5 | DENIED | 0.00 | 3 | 359 | get_order → search_policy |
| S6 | NOT_FOUND | 0.00 | 2 | 112 | get_order |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | 2781 | search_policy → search_policy → search_policy → search_policy → search_policy → search_policy |

Exactly seven normal results are saved in `outputs/agent_runs.jsonl`. Each contains status, final answer, full tool steps/observations, model calls, tokens and identifiers. The default limits are six model calls, 6,000 reported tokens and ten seconds. Monotonic time is checked at the top of each loop. Every response is charged before its tools execute; usage strictly above the limit prevents those tools. Model final claims are checked against actual observations and recorded actions.

## Fallback and interruption demonstrations

| Demonstration | Status | LLM calls | Tokens | Completed tools |
| --- | --- | --- | --- | --- |
| S1: token limit 200 | FALLBACK_TOKEN_LIMIT | 3 | 376 | 2 |
| S1: timeout 0 seconds | FALLBACK_TIMEOUT | 0 | 0 | 0 |
| S7: default iteration limit | FALLBACK_MAX_ITERATIONS | 6 | 2781 | 6 |

The token demonstration stops before calculate_refund; its violating response is counted, but its tool does not run. A post-response budget cannot prevent that response's token overshoot. Every fallback supplies a human-handoff summary with completed steps, calls and usage. Timeout zero invokes neither model nor tool.

Checkpoints under `state/` preserve full messages, tool-call IDs, finalized observations, counters, limits, elapsed active time, source hashes, pending calls and execution state. A pending intent is saved before execution and every observation is saved immediately after completion. Atomic replacement and OS file locks prevent partial writes and concurrent writers. Resume retains elapsed active time while excluding process downtime; changed source inputs or repurposed run IDs are rejected.

Resume also rejects malformed counters, changed authorization, inconsistent message histories, pending calls and cached results, and financial observations without a matching ledger event. Corrupted files are preserved for human review. This is consistency checking, not cryptographic protection against someone able to rewrite the checkpoint and ledger together. Unexpected response shapes and model/tool errors produce bounded handoff statuses rather than executing unchecked tools.

Crash simulation saves get_order → search_policy, then raises an interruption. Resume executes calculate_refund → issue_refund. The combined sequence is get_order → search_policy → calculate_refund → issue_refund; final status is REFUND for INR 128.25. No completed tool repeats, and repeat resume returns the saved result without another model call or ledger action. A separate test interrupts immediately after the financial write but before its observation: replay reconciles the pending call using the existing action ID, preventing a duplicate ledger event.

The ledger has 4 unique actions: 3 normal scenario events and 1 separate crash-demo event. There are 3 simulated issued refunds totalling INR 391.50 (including the separate INR 128.25 crash demo), and 1 escalation totalling INR 1999.00. Normal S1/S2 refunds total INR 263.25. Each entry has an actual UTC timestamp, action, order, amount, scenario, run and idempotency key. Keys use run/action/order; replays retain the original timestamp. Cross-run order deduplication is outside this simulation: the crash demo deliberately uses a different run ID for S1.

## ReAct versus plan-and-execute (under 250 words)

ReAct observes each tool result before deciding the next action. Here S1 needs five model calls and 894 tokens; a cancelled order (S5) stops after three calls and 359 tokens, while a missing order (S6) stops after two calls and 112 tokens. These measurements show useful early exits, but repeated histories add overhead. The looping S7 consumes six calls and 2,781 tokens before the iteration guard stops it.

Plan-and-execute would create a plan first and then run its steps, potentially reducing repeated model decisions. Unexpected missing or cancelled orders require explicit branches or replanning; a rigid plan must never continue to a refund after either observation. That architecture was not implemented, so no comparative token or call savings are claimed. Both approaches need validated tools, authoritative order checks, budgets, human escalation, durable checkpoints and idempotent actions. ReAct resumes the saved conversation and pending action here; a planner would need to persist plan position and observations as well. ReAct is preferable for this assignment's short, observation-dependent workflow and supplied mock. A deterministic execution plan could be attractive for a stable production process once its branches are proven.

## Artifacts and reproducibility

- `scripts/policy_rag.py`: chunking, retrieval, answering and both evaluations.
- `scripts/refund_tools.py`, `refund_agent.py`, `agent_storage.py`: tool boundaries, agent loop and durable storage; `agent_analysis.py`: scenarios and demonstrations.
- `scripts/run_part_d.py`, `validate_part_d.py`, `part_d_integrity.py`, `part_d_report.py`: orchestration, validation, A-C preservation and this report.
- `outputs/part_d/`: chunks, retrieval example, per-question evaluations, summary, demonstrations, interrupted checkpoint evidence and A-C preservation hashes.
- `outputs/part_d/pre_audit_rag_overlap_*.json`: preserved pre-fix evaluation evidence from the original implementation; fresh runs need not recreate this historical evidence.
- `outputs/agent_runs.jsonl`, `outputs/refund_ledger.jsonl`, `state/*.json`: results, simulated financial audit events and recoverable conversations.
- `tests/test_part_d.py` and the extended source-only A-through-D reproduction test.

Rerunning fixed run IDs returns finished checkpoints and preserves ledger bytes; exactly seven normal results remain. Clean reproduction uses a new source-only folder and independently executes all four parts. It compares analytical artifacts, agent decisions, counts and README; real timestamps and elapsed checkpoint durations are excluded from cross-run equality. No earlier A-C data or outputs are modified.

# Limitations and Next Steps

All data is synthetic and covers a fixed six-month window. City/category median imputation reduces observed variation; imputed delivery times are analytical estimates, not observed delivery evidence. Cancelled orders remain outside revenue and delivery analyses. Large amounts are retained because an IQR flag alone does not establish an error. Correlations are weak and do not establish causation.

SQLite REAL and pandas floats use binary arithmetic; comparisons use a tight absolute tolerance and reported currency/ranked spending is rounded to cents. For production financial accounting, use integer paise or decimal arithmetic. RFM and lapse monetary totals sum exact integer paise, reject sub-paise inputs, and serialize totals with two decimal places. Model inputs convert the exact totals to float once; probabilities and statistical means keep full precision. CSV schemas are explicit, and this README renders declared dependency pins and measured artifacts rather than the reader's Python or SQLite version.

Premium segments use the October snapshot, including September outcomes: the Premium filter contains future information, so this retention list is an illustrative post-hoc analysis, not a leakage-free prospective evaluation. Premium status and segment labels are never predictive features. The model evaluation uses the fixed clean Part A inputs; Part A's already-completed imputations are inherited, rather than refitted by Part B. A production prospective study should also fit upstream cleaning on historical training information. Parts A, B and C are complete. Part C regex redaction covers the specified synthetic PII formats; the mock and token estimator are not production evaluations. Part D is complete. TF-IDF matches vocabulary rather than meaning; an extractive context check does not prove semantic correctness, especially across chunk boundaries. Synthetic policies and a deterministic mock do not establish real-model accuracy or eliminate hallucination risk. Production refunds require authenticated customer ownership, observed delivery evidence, date eligibility and payment authorization. Local run/action idempotency needs a transactional, globally deduplicated payment ledger in production; human escalation must connect to an authenticated approval workflow. Checkpoint conversations and audit logs contain synthetic order IDs and requests; production use needs access controls, encryption, retention limits, PII minimization and tamper-evident audit storage. The time guard checks between operations and cannot interrupt a blocking model or tool call. No real payments or external messages are sent. Review the completed capstone before publication. Publishing a public GitHub repository is not performed by this local pipeline.

## Final independent audit

The audit reconstructs cleaning and SQL expectations with CSV records and Decimal arithmetic, refits K=2–8 and both full model grids independently, checks actual triage prompts and usage, and reruns the agent with fresh state. Run `python -m scripts.independent_audit`; its measured evidence is saved in `outputs/final_audit/independent_checks.json`. Adversarial regressions are in `tests/test_final_audit.py`.

Historical audit at commit 6542a6e: the supplied sources matched the original assignment. A fresh Python 3.13.7 environment installed requirements.txt successfully, and source-only reproduction outside the repository completed A–D, all validators and 135 tests with zero failures or skips. Python 3.11 syntax and package Requires-Python metadata were checked; actual execution was on Windows with Python 3.13.7, not a separate Python 3.11 or Linux/macOS runtime. SQL database and model files are intentional grading artifacts; no environment, credentials or media are tracked.

That audit fixed mixed-source RAG answers, corrupt-checkpoint handling, malformed model response handling, invalid token-budget accounting, incomplete input checks and the empty Premium retention case. It added the missing TF/IDF explanation. Normal scenario results, ledger evidence, Part A–C analytical outputs and supplied sources are unchanged. Historical raw RAG evidence remains available, while validated zero-overlap facts now score 8/12 versus the unchanged raw mock's 10/12; overlap 10 remains 12/12.

The end-to-end model evaluation is not a prospective leakage-free study: Part A explicitly imputes delivery times over the whole window. Its Personal Care median is 25.5 minutes, versus 25.0 using only pre-September observations. A counterfactual that changes September completed delivery durations to 1,000 minutes changes 450 earlier customers' average-delivery features after recleaning. This upstream limitation is retained and disclosed because the brief requires Part B to consume Part A's fixed cleaned tables; changing the production study design requires historical, training-only cleaning with preserved missingness provenance. This is distinct from the correctly isolated model encoders/scalers/resampling and the explicitly post-hoc Premium retention filter.

Per-run refund idempotency meets the assignment. The separately identified crash-demo refund is intentional evidence, not production order-level deduplication. Authentication, global payment deduplication and tamper-evident storage remain production requirements. The public submission is https://github.com/reuke1111/instamart-customer-copilot/tree/main.

The reproducibility repair pins pandas 2.3.3 and scikit-learn 1.8.0, retrains the model through Part B, fixes CSV schemas and exact monetary aggregation, and renders runtime documentation from declared pins. All original 135 tests and grading assertions are retained. Current command results, environment details, warnings and repeated-pipeline byte comparisons are recorded in `outputs/final_audit/repair_verification.json`; the earlier verification.json is historical evidence for the pre-repair commit.
