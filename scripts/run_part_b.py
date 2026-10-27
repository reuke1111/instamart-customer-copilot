"""Validate frozen Part A, execute Part B, and prove Part A was preserved."""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

from scripts import build_readme, lapse_features, lapse_models, segmentation
from scripts.part_b_integrity import record_start, assert_preserved
from scripts.validate_checkpoints import validate as validate_part_a
from scripts.validate_part_b import validate as validate_part_b
from scripts.runtime import check_environment


def main():
    if not __debug__:
        raise RuntimeError("Part B requires assertions; do not use -O")
    check_environment()
    validate_part_a()
    print("Part A: ALL CHECKPOINTS PASSED (read-only)", flush=True)
    baseline = record_start()
    segmentation.run()
    print("Segmentation: complete", flush=True)
    lapse_features.run()
    print("Lapse cohort: complete", flush=True)
    report = lapse_models.run()
    print(f"Selected {report['selected_model']}; test F1={report['models'][report['selected_model']]['test']['f1']:.6f}", flush=True)
    build_readme.run()
    assert_preserved(manifest=baseline)
    validate_part_b()
    validate_part_a()
    print("ALL PART B CHECKPOINTS PASSED")
    print("Part A artifacts unchanged")


if __name__ == "__main__":
    main()
