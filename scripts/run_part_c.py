"""Preserve Parts A/B, run Part C, render measured results and validate all."""

from scripts import build_readme, triage_analysis
from scripts.part_c_integrity import assert_preserved, record_start
from scripts.validate_checkpoints import validate as validate_a
from scripts.validate_part_b import validate as validate_b
from scripts.validate_part_c import validate as validate_c


def main():
    if not __debug__:
        raise RuntimeError("Part C requires privacy assertions; do not use -O")
    validate_a()
    validate_b()
    print("Parts A and B: checkpoints passed (read-only)", flush=True)
    baseline = record_start()
    report = triage_analysis.run()
    build_readme.run()
    assert_preserved(manifest=baseline)
    validate_c()
    validate_a()
    validate_b()
    for name, summary in report["runs"].items():
        print(f"{name}: classified={summary['classified']}, correct={summary['correct']}, tokens={summary['total_tokens']}")
    print("ALL PART C CHECKPOINTS PASSED")
    print("Part A and B artifacts unchanged")


if __name__ == "__main__":
    main()
