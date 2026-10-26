"""Preserve Parts A-C, execute Part D and validate the complete capstone."""

from scripts import agent_analysis, build_readme, policy_rag
from scripts.part_d_integrity import assert_preserved, record_start
from scripts.validate_checkpoints import validate as validate_a
from scripts.validate_part_b import validate as validate_b
from scripts.validate_part_c import validate as validate_c
from scripts.validate_part_d import validate as validate_d


def main():
    if not __debug__:
        raise RuntimeError("Part D requires validation assertions; do not use -O")
    for validate in (validate_a, validate_b, validate_c):
        validate()
    print("Parts A-C: checkpoints passed (read-only)", flush=True)
    baseline = record_start()
    policy_rag.run()
    summary, _ = agent_analysis.run()
    build_readme.run()
    assert_preserved(manifest=baseline)
    for validate in (validate_a, validate_b, validate_c, validate_d):
        validate()
    for row in summary["scenarios"]:
        print(row)
    print("ALL PART D CHECKPOINTS PASSED")
    print("Parts A-C artifacts unchanged")


if __name__ == "__main__":
    main()
