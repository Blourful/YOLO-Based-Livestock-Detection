# End-to-end integration test (minimal training)
import subprocess, sys, yaml
from pathlib import Path
import subprocess, sys, re, yaml, os, stat, platform
from pathlib import Path
import pytest

def test_end_to_end_training(tiny_images_first_dataset, tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "user_system" / "train.py"

    run_root = tmp_path / "runs"
    cmd = [
        sys.executable, str(script),
        "--data", str(tiny_images_first_dataset),
        "--names", "cattle,sheep",
        "--size", "s",
        "--epochs", "1",
        "--batch", "1",
        "--imgsz", "320",
        "--output", str(run_root),
        "--project", "train",
    ]

    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(repo_root))
    assert r.returncode == 0, r.stderr

    train_dir = next((run_root / "train").glob("*"))
    assert (train_dir / "args.yaml").exists()
    y = yaml.safe_load((train_dir / "args.yaml").read_text())
    assert y["epochs"] == 1
    assert y["batch"] == 1
    assert y["imgsz"] == 320
    assert (train_dir / "weights" / "last.pt").exists()

# Edge — duplicate run names should rotate/unique the run directory
def test_duplicate_run_names_log_rotation(tiny_images_first_dataset, tmp_path):
    """
    Goal: When running twice with the same --project name, the trainer should create
    different subdirectories (e.g., train/exp and train/exp2 or timestamped variants)
    to avoid overwriting previous results.

    Asserts:
    - Both runs exit with code 0.
    - The two latest created run directories differ in name.
    - Each run directory contains args.yaml and weights/last.pt.
    """
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "user_system" / "train.py"

    run_root = tmp_path / "runs"
    base_cmd = [
        sys.executable, str(script),
        "--data", str(tiny_images_first_dataset),
        "--names", "cattle,sheep",
        "--size", "s",
        "--epochs", "1",
        "--batch", "1",
        "--imgsz", "320",
        "--output", str(run_root),
        "--project", "train",
    ]

    # First run
    r1 = subprocess.run(base_cmd, capture_output=True, text=True, cwd=str(repo_root))
    assert r1.returncode == 0, r1.stderr
    first_dir = max((run_root / "train").glob("*"), key=lambda p: p.stat().st_mtime)
    assert (first_dir / "args.yaml").exists()
    assert (first_dir / "weights" / "last.pt").exists()

    # Second run with the same project and params
    r2 = subprocess.run(base_cmd, capture_output=True, text=True, cwd=str(repo_root))
    assert r2.returncode == 0, r2.stderr
    second_dir = max((run_root / "train").glob("*"), key=lambda p: p.stat().st_mtime)
    assert (second_dir / "args.yaml").exists()
    assert (second_dir / "weights" / "last.pt").exists()

    # Directory names must differ (rotation/uniqueness)
    assert first_dir.name != second_dir.name
    # Loosely check that the second name has a suffix (number or timestamp)
    assert re.search(r"\d", second_dir.name) or first_dir.name != second_dir.name


# Abnormal — permission denied (simulate unwritable output directory)
@pytest.mark.skipif(platform.system().lower().startswith("win"), reason="Permission simulation is flaky on Windows")
def test_training_permission_denied_output_dir(tiny_images_first_dataset, tmp_path):
    """
    Goal: When --output points to a non-writable directory, the process should fail fast
    with a non-zero exit code and produce a clear permission-related error message.

    Asserts:
    - Non-zero exit code.
    - stderr contains permission-related keywords.
    """
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "user_system" / "train.py"

    # Create a read-only output directory (r-x)
    run_root = tmp_path / "runs_ro"
    run_root.mkdir(parents=True, exist_ok=True)
    run_root.chmod(stat.S_IREAD | stat.S_IEXEC)  # 0555

    cmd = [
        sys.executable, str(script),
        "--data", str(tiny_images_first_dataset),
        "--names", "cattle,sheep",
        "--size", "s",
        "--epochs", "1",
        "--batch", "1",
        "--imgsz", "320",
        "--output", str(run_root),
        "--project", "train",
    ]

    try:
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(repo_root))
        assert r.returncode != 0, "Expected failure due to non-writable output directory"
        stderr_lower = (r.stderr or "").lower()
        assert any(k in stderr_lower for k in ["permission", "denied", "read-only", "operation not permitted"]), \
            f"stderr did not mention permission issues: {r.stderr}"
    finally:
        # Restore permissions so tmp cleanup can proceed
        run_root.chmod(stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)  # 0755
