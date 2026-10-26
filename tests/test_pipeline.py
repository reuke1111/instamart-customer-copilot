"""Reproduce from sources alone and prove validation catches corrupted artifacts."""

import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from scripts.common import ROOT
from scripts.build_readme import render
from scripts.validate_checkpoints import validate


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix=".part-a-test-", dir=ROOT)
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        for name in [
            "generate_data.py", "mock_llm.py", "source_hashes.json",
            "requirements.txt", ".gitignore", ".gitattributes",
        ]:
            shutil.copy2(ROOT / name, cls.root / name)
        for name in ["scripts", "sql"]:
            shutil.copytree(ROOT / name, cls.root / name, ignore=shutil.ignore_patterns("__pycache__"))
        cls.result = subprocess.run(
            [sys.executable, "-m", "scripts.run_part_a"], cwd=cls.root,
            capture_output=True, text=True, timeout=60,
        )
        if cls.result.returncode != 0:
            raise AssertionError(cls.result.stdout + cls.result.stderr)

    def test_independent_source_only_reproduction(self):
        self.assertIn("generated: customers=1200 products=40 orders_raw=13084 tickets=164", self.result.stdout)
        self.assertEqual(self.result.stdout.count("ALL CHECKPOINTS PASSED"), 1)
        validate(self.root)
        for folder in ["data/raw", "data/clean", "data/policies", "outputs/part_a"]:
            for generated in (self.root / folder).iterdir():
                if generated.is_file():
                    with self.subTest(path=str(generated.relative_to(self.root))):
                        expected = ROOT / generated.relative_to(self.root)
                        self.assertEqual(
                            hashlib.sha256(generated.read_bytes()).hexdigest(),
                            hashlib.sha256(expected.read_bytes()).hexdigest(),
                        )
        # Compare the entire Part A-only document, while Part B has independent
        # source-only reproduction and README checks in its own test suite.
        self.assertEqual(render(self.root, include_part_b=False), render(ROOT, include_part_b=False))
        self.assertEqual((self.root / "README.md").read_text(encoding="utf-8"), render(self.root))

    def test_direct_validator_has_exact_success_output(self):
        result = subprocess.run(
            [sys.executable, "scripts/validate_checkpoints.py"], cwd=self.root,
            capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "ALL CHECKPOINTS PASSED\n")

    def test_optimized_python_cannot_silently_skip_assertions(self):
        result = subprocess.run(
            [sys.executable, "-O", "scripts/validate_checkpoints.py"], cwd=self.root,
            capture_output=True, text=True, timeout=60,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("ALL CHECKPOINTS PASSED", result.stdout)

    def test_changed_supplied_script_is_rejected(self):
        path = self.root / "mock_llm.py"
        original = path.read_bytes()
        try:
            path.write_bytes(original + b"\n# changed\n")
            with self.assertRaisesRegex(AssertionError, "Supplied source changed"):
                validate(self.root)
        finally:
            path.write_bytes(original)

    def test_corrupted_clean_data_is_rejected(self):
        path = self.root / "data" / "clean" / "customers.csv"
        original = path.read_bytes()
        try:
            text = original.decode("utf-8").replace("Pune", "Unknown City", 1)
            self.assertNotEqual(text.encode("utf-8"), original)
            path.write_text(text, encoding="utf-8")
            with self.assertRaises(AssertionError):
                validate(self.root)
        finally:
            path.write_bytes(original)

    def test_stale_readme_is_rejected(self):
        path = self.root / "README.md"
        original = path.read_bytes()
        try:
            path.write_bytes(original + b"\nStale numbers.\n")
            with self.assertRaisesRegex(AssertionError, "README does not match"):
                validate(self.root)
        finally:
            path.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
