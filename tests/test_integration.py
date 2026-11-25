"""
End-to-end test: runs `python main.py smoke` like a user would.

Paper reference: none
Author: <name>
Adapted from: none
"""

from __future__ import annotations

import pytest


def test_main_smoke_runs(tmp_path):
    """
    Run `python main.py smoke --out-dir <tmp_path>` with subprocess.
    Check: exit code 0, result JSON exists, has all keys, test acc is in [0, 1].
    """
    pytest.skip("TODO")
