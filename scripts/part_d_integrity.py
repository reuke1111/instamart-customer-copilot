"""Read-only preservation of all completed Part A-C data and outputs."""

import hashlib
from pathlib import Path
from scripts.common import ROOT, load_json, save_json
from scripts.part_c_integrity import fingerprint as fingerprint_ab

PART_C_COMMIT = "ee9675b47e99484a3d9ce3839553e850585e2b69"


def fingerprint(root=ROOT):
    root = Path(root)
    result = fingerprint_ab(root)
    paths = [*root.joinpath("outputs/part_c").rglob("*")]
    paths += [root / "outputs" / name for name in ["triage_results.csv", "triage_budget_demo.csv"]]
    result.update({
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths) if path.is_file()
    })
    return result


def record_start(root=ROOT):
    result = {"parts_a_b_c_commit": PART_C_COMMIT, "sha256": fingerprint(root)}
    save_json(Path(root) / "outputs/part_d/parts_a_b_c_baseline.json", result)
    return result


def assert_preserved(root=ROOT, manifest=None):
    if manifest is None:
        manifest = load_json(Path(root) / "outputs/part_d/parts_a_b_c_baseline.json")
    assert manifest["parts_a_b_c_commit"] == PART_C_COMMIT
    assert fingerprint(root) == manifest["sha256"], "Part A-C artifacts changed during Part D"
