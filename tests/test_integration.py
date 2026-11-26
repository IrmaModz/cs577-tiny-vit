"""
End-to-end test: runs `python main.py smoke` like a user would.

Paper reference: none
Author: <name>, Chatgppt
Adapted from: none
"""

from __future__ import annotations

import subprocess
import sys
import json

def test_main_smoke_runs(tmp_path):
    """
    Run `python main.py smoke --out-dir <tmp_path>` with subprocess.
    Check: exit code 0, result JSON exists, has all keys, test acc is in [0, 1].
    """
    # Run the smoke command in a separate process
    result = subprocess.run(
        [sys.executable, "main.py", "smoke", "--out-dir", str(tmp_path)],
        capture_output=True,
        text=True
    )

    # Verify successful execution
    assert result.returncode == 0, result.stderr

    # Verify that the result JSON file exists
    output_path = tmp_path / "smoke_seed0.json"
    assert output_path.exists()

    # Load the saved results
    with open(output_path, "r", encoding="utf-8") as f:
        saved_results = json.load(f)

    # Verify that all required keys are present
    required_keys = {
        "experiment",
        "seed",
        "split_seed",
        "config",
        "env",
        "num_params",
        "train_time_s",
        "history",
        "test"
    }

    assert required_keys.issubset(saved_results)

    # Verify that test accuracy is within the valid range
    assert 0 <= saved_results["test"]["acc"] <= 1
