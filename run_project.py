"""Run every project pipeline twice, then audit, validate, and test the result."""

from scripts.verify_reproducibility import run


if __name__ == "__main__":
    run()
