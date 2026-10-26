"""Fingerprint the frozen Part A and B artifacts without regenerating them."""

import hashlib
from pathlib import Path
from scripts.common import ROOT, load_json, save_json
from scripts.part_b_integrity import fingerprint as fingerprint_a

PART_B_COMMIT = "8bb7f12dacef827a71754178b257bacdff24a869"


def fingerprint(root=ROOT):
    root = Path(root)
    result = fingerprint_a(root)
    paths = [*root.joinpath("outputs/part_b").rglob("*")]
    paths += [root / "outputs" / name for name in [
        "customer_segments.csv", "lapse_model.joblib", "retention_list.csv",
    ]]
    result.update({
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths) if path.is_file()
    })
    return result


def record_start(root=ROOT):
    manifest = {"parts_a_b_commit": PART_B_COMMIT, "sha256": fingerprint(root)}
    save_json(Path(root) / "outputs/part_c/parts_a_b_baseline.json", manifest)
    return manifest


def assert_preserved(root=ROOT, manifest=None):
    if manifest is None:
        manifest = load_json(Path(root) / "outputs/part_c/parts_a_b_baseline.json")
    assert manifest["parts_a_b_commit"] == PART_B_COMMIT
    assert fingerprint(root) == manifest["sha256"], "Part A or B artifacts changed during Part C"
