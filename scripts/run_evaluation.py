import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


EVALUATIONS = [
    (
        "Baseline evaluation",
        "evaluate_baselines.py",
    ),
    (
        "Retrieval similarity evaluation",
        "evaluate_retrieval.py",
    ),
    (
        "Retrieval relevance evaluation",
        "evaluate_retrieval_relevance.py",
    ),
    (
        "Resolution action extraction test",
        "test_resolution_actions.py",
    ),
    (
        "End-to-end agent evaluation",
        "evaluate_agent.py",
    ),
]


def run_script(name, script_name):
    """Run one evaluation script and report its status."""

    script_path = SCRIPTS_DIR / script_name

    if not script_path.exists():
        print(f"[ERROR] Missing script: {script_path}")
        return False

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)
    print(f"Running: {script_name}")
    print()

    start = time.time()

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PROJECT_ROOT,
        text=True,
    )

    elapsed = time.time() - start

    print()
    print(f"Completed in {elapsed:.1f} seconds.")

    if result.returncode != 0:
        print(
            f"[FAILED] {script_name} "
            f"returned exit code {result.returncode}"
        )
        return False

    print(f"[OK] {script_name}")
    return True


def main():
    print("=" * 70)
    print("SUPPORTLENS AI — LOCAL EVALUATION HARNESS")
    print("=" * 70)
    print()
    print("This harness runs the reproducible local evaluations.")
    print("It does not call the Gemini API.")
    print()

    start = time.time()

    results = []

    for name, script_name in EVALUATIONS:
        success = run_script(name, script_name)
        results.append((name, success))

    total_time = time.time() - start

    print()
    print("=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    for name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{status:>5}  {name}")

    print()
    print(f"Total runtime: {total_time:.1f} seconds")

    failed = sum(
        1 for _, success in results
        if not success
    )

    print()

    if failed:
        print(
            f"[WARNING] {failed} evaluation(s) failed."
        )
        sys.exit(1)

    print("All local evaluations completed successfully.")


if __name__ == "__main__":
    main()