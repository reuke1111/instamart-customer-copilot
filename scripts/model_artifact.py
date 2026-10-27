"""Version-checked loading of this project's trusted local model only."""

import hashlib
from pathlib import Path
import warnings

import joblib
from sklearn.exceptions import InconsistentVersionWarning

from scripts.common import ROOT, load_json, save_json
from scripts.runtime import check_environment, require_versions


def save_model(model, root=ROOT):
    root = Path(root)
    dependencies = check_environment(root)
    path = root / "outputs/lapse_model.joblib"
    joblib.dump(model, path, compress=3, protocol=4)
    save_json(root / "outputs/part_b/model_metadata.json", {
        "schema_version": 1,
        "dependencies": dependencies,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    })


def load_model(root=ROOT):
    root = Path(root)
    metadata = load_json(root / "outputs/part_b/model_metadata.json")
    if metadata.get("schema_version") != 1:
        raise ValueError("Unsupported model metadata schema")
    dependencies = check_environment(root)
    if metadata.get("dependencies") != dependencies:
        raise RuntimeError("Model training dependencies differ from requirements.txt; "
                           "regenerate with python -m scripts.run_part_b")
    require_versions(metadata["dependencies"])
    path = root / "outputs/lapse_model.joblib"
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata.get("sha256"):
        raise ValueError("Model checksum mismatch; regenerate through the Part B pipeline")
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        return joblib.load(path)
