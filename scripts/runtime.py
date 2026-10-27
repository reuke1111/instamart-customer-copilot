"""Check the pinned environment before generating or loading model artifacts."""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def pinned_versions(root=ROOT):
    return dict(line.strip().split("==", 1) for line in
                (Path(root) / "requirements.txt").read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.startswith("#"))


def installed_versions(packages):
    result = {}
    for name in packages:
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = "not installed"
    return result


def require_versions(expected):
    actual = installed_versions(expected)
    mismatches = [f"{name}: expected {wanted}, found {actual[name]}"
                  for name, wanted in expected.items() if actual[name] != wanted]
    if mismatches:
        raise RuntimeError("Dependency mismatch; " + "; ".join(mismatches) +
                           ". Use the project virtual environment and run "
                           "python -m pip install -r requirements.txt. "
                           "Do not load models across scikit-learn versions.")
    return actual


def check_environment(root=ROOT):
    return require_versions(pinned_versions(root))


if __name__ == "__main__":
    check_environment()
    print("PINNED ENVIRONMENT VERIFIED")
