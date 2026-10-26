"""Render every required result into README from measured Part A artifacts."""

from pathlib import Path
import platform
import sqlite3

import numpy as np
import pandas as pd

from scripts.common import ROOT, load_json


def table(headers, rows):
    def cell(value):
        return str(value).replace("|", r"\|").replace("\n", " ")
    return "\n".join([
        "| " + " | ".join(map(cell, headers)) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(map(cell, row)) + " |" for row in rows),
    ])


def number(value):
    return f"{value:,.2f}"


def eda_answers(stats, relationship):
    return [
        "**What is this data about?** Synthetic Instamart orders across six cities "
        "and five product categories from April to September 2025. Revenue and "
        "delivery analysis use completed orders only.",
        "**What do its distributions and statistics look like?** Amounts are "
        f"right-skewed: mean INR {number(stats['mean'])}, median INR {number(stats['median'])}, "
        f"and skewness {stats['skewness']:.2f}. The middle half lies between "
        f"INR {number(stats['q1'])} and INR {number(stats['q3'])}.",
        "**Are there outliers or anomalies?** "
        f"{stats['upper_outliers']} completed amounts exceed the upper IQR fence of "
        f"INR {number(stats['upper_fence'])}; {stats['lower_outliers']} fall below the lower fence. "
        "High-priced electronics can explain large valid purchases, so outliers are retained. "
        "Raw duplicates, earlier pending snapshots, invalid ages, amounts and durations "
        "are corrected using the documented rules.",
        "**What relationships exist between variables?** "
        f"The largest absolute Pearson correlation is {relationship['left']} with "
        f"{relationship['right']} ({relationship['coefficient']:.6f}), a weak negative association. "
        "The amount formula includes discount, while product prices and quantities also vary. "
        "Correlation of this size does not establish causation.",
    ]


def render(root=ROOT, include_part_b=True):
    root = Path(root)
    folder = root / "outputs" / "part_a"
    audit = load_json(folder / "audit.json")
    cleaning = load_json(folder / "cleaning.json")
    sql = load_json(folder / "sql_summary.json")
    eda = load_json(folder / "eda.json")
    pivot = pd.read_csv(folder / "revenue_pivot.csv").set_index("city")
    city = pd.read_csv(folder / "city_aggregates.csv").set_index("city")
    correlations = pd.read_csv(folder / "correlations.csv", index_col=0)
    summary, stats = cleaning["summary"], eda["statistics"]
    answers = eda_answers(stats, eda["largest_absolute_correlation"])
    assert sum(len(answer.split()) for answer in answers) <= 200
    lines = [
        "# Overview", "",
        "Part A of the Instamart Customer Intelligence & Support Copilot capstone: "
        "audit, clean, store and query deterministic synthetic data. All monetary values "
        "are in INR. This is fictional data and does not describe real Instamart operations. "
        "Parts B, C and D are not implemented.", "",
        "The supplied scripts are preserved byte-for-byte from the assignment. "
        "`source_hashes.json` records their SHA-256 hashes; validation checks both. "
        "The generator also writes tickets, policy files and evaluation scenarios as "
        "required by its unmodified source; producing those fixtures does not implement "
        "the later Parts.", "",
        "# Reproduce", "",
        "Prerequisites: Python 3.11+ (the pinned dependencies were tested with "
        f"Python {platform.python_version()}), Git, and a shell at the repository root "
        "on branch `main`. Install dependencies once; all subsequent steps run offline. "
        "Run commands in this order (activation below is PowerShell):", "",
        "1. `python -m venv .venv`",
        "2. `.\\.venv\\Scripts\\Activate.ps1`",
        "3. `python -m pip install -r requirements.txt`",
        "4. `python -m pip check`",
        "5. `python generate_data.py`",
        "6. `python -m scripts.audit_data`",
        "7. `python -m scripts.clean_data`",
        "8. `python -m scripts.sql_analysis`",
        "9. `python -m scripts.exploratory_analysis`",
        "10. `python -m scripts.build_readme`",
        "11. `python scripts/validate_checkpoints.py`",
        "12. `python -m unittest discover -s tests -v`", "",
        "On macOS/Linux replace step 2 with `source .venv/bin/activate`. "
        "If PowerShell activation is unavailable, use `.venv\\Scripts\\python.exe` "
        "in place of `python` for steps 3–12. The local Windows Python alias was "
        "unavailable during setup, so `py -3.13 -m venv .venv` created the environment. "
        "`requirements.txt` was produced with `python -m pip freeze`; pandas and NumPy "
        "are the only direct third-party dependencies. python-dateutil, six and tzdata "
        "are pandas dependencies.", "",
        "Equivalent full regeneration after installation: `python -m scripts.run_part_a`, "
        "then `python -m unittest discover -s tests -v`. "
        "The orchestrator verifies source hashes before running the generator. "
        "The validator reads artifacts without rewriting them and prints exactly "
        "`ALL CHECKPOINTS PASSED` on success. Do not use `python -O`, which disables assertions.", "",
        f"Measured runtime: Python {platform.python_version()}, pandas {pd.__version__}, "
        f"NumPy {np.__version__}, SQLite {sqlite3.sqlite_version}.", "",
        "# Part A", "",
        "## Generation and raw audit", "",
        "Generator output: `generated: customers=1200 products=40 orders_raw=13084 tickets=164`.", "",
        "Missing values below use pandas' default CSV NA recognition: the literal "
        "`N/A` is missing, while `error` remains text until numeric coercion. "
        "Thus 53 raw amounts are already missing on load, but 120 amounts are "
        "non-numeric in total. Raw age, gender and duration blanks are counted separately. "
        "Tickets are not processed in Part A; any verification read uses `keep_default_na=False`.", "",
    ]
    for name, info in audit["tables"].items():
        lines += [f"**{name}: {info['rows']:,} rows × {info['columns']} columns**", "",
                  table(["Column", "pandas dtype", "Missing count", "Missing %"], [
                      [row["column"], row["dtype"], row["missing_count"], f"{row['missing_pct']:.4f}"]
                      for row in info["column_audit"]
                  ]), ""]
    lines += [
        table(["Audit metric", "Measured value"], [
            [name.replace("_", " "), f"{value:,}"] for name, value in audit["metrics"].items()
        ]), "",
        "## Cleaning rules and changes", "",
        "Operations run in the order shown. Counts for parsing denote values converted "
        "to a new type; counts for replacement and imputation denote values changed; "
        "duplicate counts denote rows removed. Customer IDs are preserved exactly, "
        "and the product catalogue is unchanged.", "",
        table(["Customer operation", "Values changed or parsed"], [
            [row["rule"], f"{row['changed']:,}"] for row in cleaning["customer_rules"]
        ]), "",
        "Ages outside 10–100 are replaced before computing each normalized city's median "
        "with `groupby('city')['age'].transform('median')`. The 61 original blanks and "
        "6 invalid ages make 67 imputations. No global median or rounded age is substituted.", "",
        table(["City", "Customers", "Age median used"], [
            [name, summary["city_counts"][name], value]
            for name, value in cleaning["city_age_medians"].items()
        ]), "",
        table(["Gender", "Customers"], summary["gender_counts"].items()), "",
        f"Final ages: {summary['missing_ages']} missing; minimum {summary['min_age']:.0f}, "
        f"maximum {summary['max_age']:.0f}. All {summary['customers']:,} identifiers survive unchanged.", "",
        table(["Order operation", "Rows/values changed or parsed"], [
            [row["rule"], f"{row['changed']:,}"] for row in cleaning["order_rules"]
        ]), "",
    ]
    conversion = cleaning["order_rules"][3]
    cancelled = cleaning["order_rules"][-1]
    lines += [
        f"After exact duplicates and earlier snapshots are removed, "
        f"{conversion['changed']:,} nonmissing amount strings are converted: "
        f"{conversion['new_missing']} become newly missing, while "
        f"{conversion['already_missing']} were already missing (`N/A`). "
        f"All {conversion['invalid_total']} missing amounts are recovered with "
        "`round(price_inr * quantity * (1 - discount_pct / 100), 2)`.", "",
        "Negative delivery times are made missing before category medians are calculated. "
        "Only completed orders participate in `groupby(category).transform('median')`. "
        "Of the 774 completed imputations, 669 were already blank and 105 were negative. "
        f"Cancelled orders receive no imputation: {cancelled['changed']} nonmissing values "
        f"are cleared and {cancelled['left_missing']:,} remain missing. "
        "Conflicting snapshots at the same timestamp, unknown products, pending final "
        "snapshots and groups without valid imputation observations raise errors.", "",
        table(["Product category", "Completed delivery median (minutes)"],
              cleaning["category_delivery_medians"].items()), "",
        table(["Final checkpoint", "Measured value"], [
            ["Orders", f"{summary['orders']:,}"],
            ["Completed", f"{summary['completed']:,}"],
            ["Cancelled", f"{summary['cancelled']:,}"],
            ["Pending", summary["pending"]],
            ["Missing amounts", summary["missing_amounts"]],
            ["Missing completed delivery times", summary["missing_completed_delivery"]],
            ["Missing cancelled delivery times", f"{summary['missing_cancelled_delivery']:,}"],
            ["Completed revenue (INR)", number(summary["completed_revenue"])],
        ]), "",
        "## Storage and automated validation", "",
        "`data/clean/customers.csv`, `products.csv` and `orders.csv` contain the clean "
        "tables. `data/instamart.db` contains `customers`, `products` and `orders`, "
        "with primary keys, foreign keys and domain checks. Both signup dates and "
        "order timestamps use `YYYY-MM-DD HH:MM:SS`; cancelled durations are SQL NULL. "
        "A complete temporary SQLite file replaces the previous database only after "
        "successful insertion. Re-running the stages replaces outputs rather than appending.", "",
        "`scripts/validate_checkpoints.py` asserts all deterministic Part A reference "
        "counts, sums, rankings and EDA checkpoints, with explicit tolerances only for "
        "rounded or approximate statistics. It also checks raw-to-clean transformation, "
        "CSV/SQLite agreement, keys, NULLs, timestamps, all saved reports and this README. "
        "Standard-library unittest cases cover cleaning edge cases, SQL semantics, "
        "SQLite round-trips, independent clean-folder reproduction and detection of "
        "tampered outputs. Test failures are not suppressed.", "",
        "## Seven SQL queries", "",
        "All statements are in `sql/queries.sql`; each runs using `pd.read_sql_query`. "
        "Full result sets are saved as `outputs/part_a/q1.csv` through `q7.csv`. "
        "Revenue/spend includes only completed orders. Q3 uses the full catalogue.", "",
        table(["Query", "Measured result"], [
            ["Q1 WHERE then HAVING", f"{sql['row_counts']['Q1']} customers: individual amounts > INR 500; their subtotal > INR 8,000"],
            ["Q2 Three-table join", f"{sql['row_counts']['Q2']} cells; INR {number(sql['q2_total'])}; highest Bengaluru / Electronics, INR {number(sql['q2_highest']['revenue'])}"],
            ["Q3 Scalar subquery", f"{sql['row_counts']['Q3']} products above average catalogue price"],
            ["Q4 LEFT JOIN", f"{sql['row_counts']['Q4']} customers with no completed orders"],
            ["Q5 Two CTEs", f"{sql['row_counts']['Q5']} customers above mean completed spending of INR {number(sql['q5_average'])}"],
            ["Q6 Ranking windows", f"{sql['row_counts']['Q6']:,} customers with completed orders; {sql['q6_rank_row_number_differences']} rows where RANK differs from ROW_NUMBER"],
            ["Q7 Cumulative SUM", f"{sql['row_counts']['Q7']} days; final INR {number(sql['q7_final_total'])}"],
        ]), "",
        "Q5 averages the 1,175 customers with completed orders; customers with no completed "
        "orders are not added as zero spenders. Spending totals are rounded to cents "
        "before ranking so floating-point summation cannot split monetary ties. "
        "Ties share RANK (with subsequent gaps) and DENSE_RANK (without gaps), while "
        "ROW_NUMBER assigns a distinct position to every row, explaining the eight differences. "
        "Within a tied group ROW_NUMBER order is not a business distinction.", "",
        table(["City", "Top customer", "Completed spend (INR)"], [
            [row["city"], row["customer_id"], number(row["total_spend"])]
            for row in sql["q6_leaders"]
        ]), "",
        "## pandas cross-check and EDA", "",
        "The following `pivot_table(margins=True)` reproduces all 30 SQL Q2 cells "
        "(checked individually as well as in total). All values are INR.", "",
        table(["City", *pivot.columns], [
            [name, *[number(value) for value in row]]
            for name, row in pivot.iterrows()
        ]), "",
        f"SQL grand total = pandas grand total = INR {number(eda['pivot_grand_total'])}.", "",
        "Named aggregation uses `revenue_sum=('order_amount', 'sum')`, "
        "`revenue_mean=('order_amount', 'mean')` and "
        "`completed_order_count=('order_amount', 'count')`.", "",
        table(["City", "Revenue sum (INR)", "Mean amount (INR)", "Completed count"], [
            [name, number(row["revenue_sum"]), number(row["revenue_mean"]), int(row["completed_order_count"])]
            for name, row in city.iterrows()
        ]), "",
        table(["Completed amount statistic", "Measured value"], [
            ["Mean (INR)", number(stats["mean"])],
            ["Median / Q2 (INR)", number(stats["median"])],
            ["Skewness (pandas unbiased sample skew)", f"{stats['skewness']:.6f}"],
            ["Q1 (INR)", number(stats["q1"])], ["Q3 (INR)", number(stats["q3"])],
            ["IQR (INR)", number(stats["iqr"])],
            ["Lower 1.5 x IQR fence (INR)", number(stats["lower_fence"])],
            ["Upper 1.5 x IQR fence (INR)", number(stats["upper_fence"])],
            ["Below lower fence", stats["lower_outliers"]],
            ["Above upper fence", stats["upper_outliers"]],
        ]), "",
        "Pearson correlation matrix, over all 11,110 completed orders after cleaning:", "",
        table(["Variable", *correlations.columns], [
            [name, *[f"{value:.6f}" for value in row]] for name, row in correlations.iterrows()
        ]), "",
        "**Four EDA answers (under 200 words total)**", "",
        *[item for answer in answers for item in (answer, "")],
        "## Artifact map", "",
        "- Supplied sources: `generate_data.py`, `mock_llm.py`, `source_hashes.json`.",
        "- Modular stages: `scripts/audit_data.py`, `clean_data.py`, `sql_analysis.py`, "
        "`exploratory_analysis.py`, `build_readme.py`, `validate_checkpoints.py`, `run_part_a.py`.",
        "- Shared helpers and fixed references: `scripts/common.py`, `scripts/checkpoints.py`.",
        "- Audit/cleaning/SQL/EDA JSON reports, seven query CSVs, pivot, city aggregates "
        "and correlation matrix: `outputs/part_a/`.",
        "- Raw and clean CSVs: `data/raw/`, `data/clean/`; SQLite: `data/instamart.db`.",
        "- Tests: `tests/`; dependencies: `requirements.txt`; Git exclusions: `.gitignore`; "
        "source-byte preservation: `.gitattributes`.", "",
        "# Part B", "", "Not yet implemented. Requires separate approval after Part A.", "",
        "# Part C", "", "Not yet implemented. Requires separate approval after Part A.", "",
        "# Part D", "", "Not yet implemented. Requires separate approval after Part A.", "",
        "# Limitations and Next Steps", "",
        "All data is synthetic and covers a fixed six-month window. City/category "
        "median imputation reduces observed variation; imputed delivery times are "
        "analytical estimates, not observed delivery evidence. Cancelled orders remain "
        "outside revenue and delivery analyses. Large amounts are retained because an "
        "IQR flag alone does not establish an error. Correlations are weak and do not "
        "establish causation.", "",
        "SQLite REAL and pandas floats use binary arithmetic; comparisons use a tight "
        "absolute tolerance and reported currency/ranked spending is rounded to cents. "
        "For production financial accounting, use integer paise or decimal arithmetic. "
        "Raw-string dtype labels and formatting can vary across pandas versions, so "
        "the measured environment is pinned and README results are generated from outputs.", "",
        "Next step: review Part A and approve any work on Parts B–D. No segmentation, "
        "prediction, ticket triage, policy assistant or refund agent has been implemented. "
        "Publishing a public GitHub repository is not performed by this local pipeline.", "",
    ]
    content = "\n".join(lines)
    if include_part_b:
        from scripts.part_b_report import extend_readme
        content = extend_readme(content, root)
        from scripts.part_c_report import extend_readme as extend_part_c
        content = extend_part_c(content, root)
    return content


def run(root=ROOT):
    path = Path(root) / "README.md"
    path.write_text(render(root), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    run()
