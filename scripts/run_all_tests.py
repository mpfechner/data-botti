

# scripts/run_all_tests.py
import os
import sys
import subprocess
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
_LOG_DIR = os.path.join(os.path.dirname(_HERE), "logs")
_LOG_FILE = os.path.join(_LOG_DIR, "test_run.log")


def main():
    os.makedirs(_LOG_DIR, exist_ok=True)
    test_files = [
        f for f in os.listdir(_HERE)
        if f.endswith(".py") and (
            f.startswith("test_") or f.endswith("_smoketest.py") or f.endswith("_checks.py")
        )
    ]

    results = []
    with open(_LOG_FILE, "w", encoding="utf-8") as log:
        log.write(f"=== Test Run started {datetime.utcnow().isoformat()}Z ===\n")

        for f in sorted(test_files):
            path = os.path.join(_HERE, f)
            log.write(f"\n--- Running {f} ---\n")
            print(f"[RUN] {f}")
            proc = subprocess.run([sys.executable, path], capture_output=True, text=True)
            if proc.returncode == 0:
                results.append((f, "PASS"))
                log.write(proc.stdout)
                log.write("[RESULT] PASS\n")
            else:
                results.append((f, "FAIL"))
                log.write(proc.stdout)
                log.write(proc.stderr)
                log.write("[RESULT] FAIL\n")
            log.flush()

        log.write("\n=== Summary ===\n")
        for name, res in results:
            log.write(f"{name}: {res}\n")
        log.write(f"=== Test Run finished {datetime.utcnow().isoformat()}Z ===\n")

    # Print summary to console
    print("\n=== Test Summary ===")
    for name, res in results:
        print(f"{name}: {res}")

    failed = [r for _, r in results if r == "FAIL"]
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()