# Overview

Part A of the Instamart Customer Intelligence & Support Copilot capstone: audit, clean, store and query deterministic synthetic data. All monetary values are in INR. This is fictional data and does not describe real Instamart operations. Parts B, C and D are not implemented.

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
12. `python -m unittest discover -s tests -v`

On macOS/Linux replace step 2 with `source .venv/bin/activate`. If PowerShell activation is unavailable, use `.venv\Scripts\python.exe` in place of `python` for steps 3–12. The local Windows Python alias was unavailable during setup, so `py -3.13 -m venv .venv` created the environment. `requirements.txt` was produced with `python -m pip freeze`; pandas and NumPy are the only direct third-party dependencies. python-dateutil, six and tzdata are pandas dependencies.

Equivalent full regeneration after installation: `python -m scripts.run_part_a`, then `python -m unittest discover -s tests -v`. The orchestrator verifies source hashes before running the generator. The validator reads artifacts without rewriting them and prints exactly `ALL CHECKPOINTS PASSED` on success. Do not use `python -O`, which disables assertions.

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

Not yet implemented. Requires separate approval after Part A.

# Part C

Not yet implemented. Requires separate approval after Part A.

# Part D

Not yet implemented. Requires separate approval after Part A.

# Limitations and Next Steps

All data is synthetic and covers a fixed six-month window. City/category median imputation reduces observed variation; imputed delivery times are analytical estimates, not observed delivery evidence. Cancelled orders remain outside revenue and delivery analyses. Large amounts are retained because an IQR flag alone does not establish an error. Correlations are weak and do not establish causation.

SQLite REAL and pandas floats use binary arithmetic; comparisons use a tight absolute tolerance and reported currency/ranked spending is rounded to cents. For production financial accounting, use integer paise or decimal arithmetic. Raw-string dtype labels and formatting can vary across pandas versions, so the measured environment is pinned and README results are generated from outputs.

Next step: review Part A and approve any work on Parts B–D. No segmentation, prediction, ticket triage, policy assistant or refund agent has been implemented. Publishing a public GitHub repository is not performed by this local pipeline.
