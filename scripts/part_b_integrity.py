"""Read-only fingerprinting of Part A inputs and verified analytical artifacts."""

import hashlib
from pathlib import Path
from scripts.common import ROOT, load_json, save_json

PART_A_COMMIT = "0dc9d14b996f52b614115a680b75a94b9ccb3468"


def fingerprint(root=ROOT):
    root = Path(root)
    paths = [*root.joinpath("data").rglob("*"), *root.joinpath("outputs/part_a").rglob("*")]
    paths += [root / name for name in ["generate_data.py", "mock_llm.py", "sql/queries.sql"]]
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths) if path.is_file()
    }


def record_start(root=ROOT):
    path = Path(root) / "outputs/part_b/part_a_baseline.json"
    previous = load_json(path) if path.exists() else {}
    manifest = {
        "part_a_commit": PART_A_COMMIT,
        "initial_working_tree_status": previous.get("initial_working_tree_status", "source-only reproduction"),
        "sha256": fingerprint(root),
    }
    save_json(path, manifest)
    return manifest


def assert_preserved(root=ROOT, manifest=None):
    if manifest is None:
        manifest = load_json(Path(root) / "outputs/part_b/part_a_baseline.json")
    assert manifest["part_a_commit"] == PART_A_COMMIT
    assert fingerprint(root) == manifest["sha256"], "Part A artifacts changed during Part B"
