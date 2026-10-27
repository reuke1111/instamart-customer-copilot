"""Regenerate all Part A data, reports and README, then validate."""

import subprocess
import sys

from scripts import audit_data, build_readme, clean_data, exploratory_analysis, sql_analysis
from scripts.common import ROOT, verify_sources
from scripts.validate_checkpoints import validate
from scripts.runtime import check_environment


def main():
    if not __debug__:
        raise RuntimeError("Part A requires assertions; do not run Python with -O")
    check_environment()
    verify_sources()
    result = subprocess.run(
        [sys.executable, str(ROOT / "generate_data.py")],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    expected = "generated: customers=1200 products=40 orders_raw=13084 tickets=164"
    assert result.stdout.strip() == expected, result.stdout
    print(result.stdout.strip())
    for name, stage in [
        ("Audit", audit_data), ("Cleaning and storage", clean_data),
        ("SQL analysis", sql_analysis), ("EDA", exploratory_analysis),
        ("README", build_readme),
    ]:
        stage.run()
        print(f"{name}: complete")
    validate()
    print("ALL CHECKPOINTS PASSED")


if __name__ == "__main__":
    main()
