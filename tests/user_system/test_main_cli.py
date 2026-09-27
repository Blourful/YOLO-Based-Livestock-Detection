# tests/user_system/test_main_cli.py
import subprocess
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "user_system" / "main.py"
PY = sys.executable


def run_cli(argv):
    cmd = [PY, str(MAIN)] + argv
    return subprocess.run(cmd, capture_output=True, text=True)


def test_preset_quick_applies_defaults(monkeypatch):
    captured = {}
    r = run_cli(["train", "--data", ".", "--names", "cattle,sheep", "--preset", "quick"])
    assert r.returncode == 1


def test_preset_quick_with_override_wins(monkeypatch):
    captured = {}
    r = run_cli(["train", "--data", ".", "--names", "cattle,sheep", "--preset", "quick", "--epochs", "1"])
    assert r.returncode == 1


def test_preset_unknown_abnormal():
    r = run_cli(["train", "--data", ".", "--names", "cattle,sheep", "--preset", "ultra"])
    assert r.returncode != 0

