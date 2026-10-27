"""Run A-D twice, require byte stability, then audit, validate and test.

Run with the project's pinned Python environment. Every command is executed;
no success evidence is written until all commands and comparisons pass.
"""

import hashlib
from pathlib import Path
import platform
import re
import subprocess
import sys

from scripts.common import ROOT, save_json, verify_sources
from scripts.runtime import check_environment


def snapshot(root=ROOT):
    root = Path(root)
    paths = [root / "README.md"]
    for folder in ("data", "outputs", "state"):
        paths.extend((root / folder).rglob("*"))
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths) if p.is_file() and p.suffix not in (".lock", ".tmp")
            and p.name != "repair_verification.json" and not p.name.startswith("reproduction_python_")}


def run(root=ROOT):
    root = Path(root)
    dependencies = check_environment(root)
    sources = verify_sources(root)
    commands = []

    def execute(arguments):
        display = "python " + " ".join(arguments)
        print(display, flush=True)
        result = subprocess.run([sys.executable, *arguments], cwd=root,
                                capture_output=True, text=True, timeout=300)
        output = result.stdout + result.stderr
        warnings = sorted(set(re.findall(r"\b\w*Warning: [^\n]+", output)))
        commands.append({"command": display, "exit_code": result.returncode, "warnings": warnings})
        if result.returncode:
            print(output, flush=True)
            raise RuntimeError(f"Verification failed: {display}")
        print("  passed", flush=True)
        return output

    execute(["-m", "pip", "check"])
    execute(["-m", "scripts.runtime"])
    before = None
    for cycle in (1, 2):
        print(f"Complete pipeline pass {cycle}", flush=True)
        for part in "abcd":
            execute(["-m", f"scripts.run_part_{part}"])
        execute(["-m", "scripts.build_readme"])
        execute(["-m", "scripts.independent_audit"])
        current = snapshot(root)
        if before is not None:
            drift = sorted(path for path in before.keys() | current.keys()
                           if before.get(path) != current.get(path))
            if drift:
                raise AssertionError(f"Repeated pipeline changed artifacts: {drift}")
        before = current
    for validator in ("validate_checkpoints", "validate_part_b", "validate_part_c", "validate_part_d"):
        execute([f"scripts/{validator}.py"])
    output = execute(["-m", "unittest", "discover", "-s", "tests", "-v"])
    count = int(re.search(r"Ran (\d+) tests? in", output).group(1))
    if not output.rstrip().endswith("OK") or count < 135:
        raise AssertionError("Full suite must include at least the original 135 tests, with no skips")
    if snapshot(root) != before:
        raise AssertionError("Validators or tests changed persisted artifacts")
    report = {
        "status": "PASSED", "python": platform.python_version(), "platform": platform.system(),
        "dependencies": dependencies, "source_sha256": sources,
        "commands": commands, "tests": {"run": count, "failures": 0, "errors": 0, "skipped": 0},
        "repeat_pipeline": {"passes": 2, "files_compared": len(before), "drift": [], "sha256": before},
        "warnings": sorted(set(w for command in commands for w in command["warnings"])),
    }
    save_json(root / f"outputs/final_audit/reproduction_python_{sys.version_info.major}_{sys.version_info.minor}.json", report)
    print(f"REPRODUCIBILITY VERIFIED: {count} tests, {len(before)} stable files, zero failures/errors/skips", flush=True)
    return report


if __name__ == "__main__":
    run()
