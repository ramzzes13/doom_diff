"""Master script: Run the entire MemGameNGen pipeline end-to-end."""

import sys
import os
import subprocess
import time

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(PROJECT_DIR, "scripts")


def run_script(script_name: str, description: str) -> bool:
    """Run a Python script and return success status."""
    print(f"\n{'='*60}")
    print(f"STAGE: {description}")
    print(f"Script: {script_name}")
    print(f"{'='*60}\n")

    script_path = os.path.join(SCRIPTS_DIR, script_name)
    start_time = time.time()

    result = subprocess.run(
        [sys.executable, script_path],
        cwd=PROJECT_DIR,
        capture_output=False,
    )

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"STAGE '{description}' {'COMPLETED' if result.returncode == 0 else 'FAILED'}")
    print(f"Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'='*60}\n")

    return result.returncode == 0


def main():
    print("="*60)
    print("MemGameNGen: Full Pipeline Execution")
    print("="*60)
    print(f"Project dir: {PROJECT_DIR}")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    stages = [
        ("01_train_ppo.py", "Train PPO Agent"),
        ("02_collect_data.py", "Collect Training Data"),
        ("03_train_diffusion.py", "Train Diffusion Model"),
        ("04_evaluate.py", "Evaluate Model"),
    ]

    results = {}
    total_start = time.time()

    for script, desc in stages:
        success = run_script(script, desc)
        results[desc] = "SUCCESS" if success else "FAILED"

        if not success:
            print(f"\nWARNING: Stage '{desc}' failed. Continuing with next stage...")

    total_time = time.time() - total_start

    print("\n" + "="*60)
    print("PIPELINE SUMMARY")
    print("="*60)
    for stage, status in results.items():
        print(f"  {stage}: {status}")
    print(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print("="*60)


if __name__ == "__main__":
    main()
