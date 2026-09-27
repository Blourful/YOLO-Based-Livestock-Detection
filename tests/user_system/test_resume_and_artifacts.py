import json
from pathlib import Path
import pytest

from user_system.main import build_parser, run_train


def _args(argv):
    return build_parser().parse_args(argv)


def _mk_ckpt(dirpath: Path):
    ckpt = dirpath / "weights" / "last.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_bytes(b"\x00")
    return ckpt


def test_artifact_inventory_light(tmp_path, capsys, monkeypatch, install_fake_dataset):
    runs = tmp_path / "runs"
    runs.mkdir(parents=True, exist_ok=True)

    def fake_impl_train(**kw):
        run_dir = kw["run_dir"]
        (run_dir / "weights").mkdir(parents=True, exist_ok=True)
        (run_dir / "weights" / "last.pt").write_bytes(b"0")
        (run_dir / "args.yaml").write_text("epochs: 1\n")
        (run_dir / "metrics.json").write_text(json.dumps({"epochs": 1}))
        (run_dir / "results.png").write_bytes(b"\x89PNG\r\n")
        (run_dir / "logs.txt").write_text("ok\n")
        (run_dir / "images_vis").mkdir(parents=True, exist_ok=True)
        print(f"[train] run_dir={run_dir}")
        return 0

    monkeypatch.setattr("user_system.main._impl_train", fake_impl_train, raising=False)

    args = _args(["train", "--data", ".", "--names", "cattle,sheep", "--epochs", "1", "--output", str(runs), "--device", "cpu"])
    run_train(args)

    run_roots = sorted((runs / "train").glob("*"))
    assert run_roots, "No run directory created"
    run_dir = run_roots[-1]
    assert (run_dir / "args.yaml").exists()
    assert (run_dir / "weights" / "last.pt").exists()
    assert sum((run_dir / p).exists() for p in ["metrics.json", "results.png", "images_vis", "logs.txt"]) >= 2


def test_resume_vs_fresh(tmp_path, capsys, monkeypatch, install_fake_dataset):
    root = tmp_path / "runs" / "exp1"
    root.mkdir(parents=True, exist_ok=True)
    _mk_ckpt(root)

    def fake_impl_train(**kw):
        print(f"[train] resume_from={kw.get('resume_from')}")
        return 0

    monkeypatch.setattr("user_system.main._impl_train", fake_impl_train, raising=False)

    # 1) Resume path: should print a non-None checkpoint path (at least the key is present)
    args_resume = _args([
        "train", "--data", ".", "--names", "cattle,sheep", "--epochs", "1",
        "--resume", "--output", str(root), "--device", "cpu"
    ])
    run_train(args_resume)
    out_resume = capsys.readouterr().out
    assert "resume_from=" in out_resume

    # 2) Fresh path (no --resume): printed value can be None or False depending on app
    args_fresh = _args([
        "train", "--data", ".", "--names", "cattle,sheep", "--epochs", "1",
        "--output", str(root), "--device", "cpu"
    ])
    run_train(args_fresh)
    out_fresh = capsys.readouterr().out
    assert ("resume_from=None" in out_fresh) or ("resume_from=False" in out_fresh)

def test_edge_resume_prefers_last_checkpoint_over_intermediate(tmp_path, capsys, monkeypatch, install_fake_dataset):
    """
    Edge — resume after partial epoch: prefer 'weights/last.pt' even if intermediate
    checkpoints or partial markers exist; fresh run should ignore checkpoints.
    """
    runs = tmp_path / "runs" / "train" / "exp1"
    (runs / "weights").mkdir(parents=True, exist_ok=True)
    # Create an intermediate checkpoint and a partial marker
    (runs / "weights" / "epoch_0002.pt").write_bytes(b"\x01\x02")
    (runs / "partial.state").write_text("epoch=2")
    # Create canonical 'last.pt'
    last_ckpt = _mk_ckpt(runs)

    def fake_impl_train(**kw):
        # Ensure the trainer chooses 'last.pt' for resume
        resume_from = kw.get("resume_from")
        print(f"[train-edge] resume_from={resume_from}")
        assert resume_from is not None and str(resume_from).endswith("last.pt")
        return 0

    monkeypatch.setattr("user_system.main._impl_train", fake_impl_train, raising=False)

    # Resume path should pick last.pt
    args = _args([
        "train", "--data", ".", "--names", "cattle,sheep", "--epochs", "1",
        "--resume", "--output", str(runs), "--device", "cpu"
    ])
    run_train(args)
    out = capsys.readouterr().out
    assert "resume_from=" in out and str(last_ckpt) in out


def test_abnormal_corrupt_or_mismatched_checkpoint(tmp_path, capsys, monkeypatch, install_fake_dataset):
    """
    Abnormal — corrupt or mismatched checkpoint handling: when the selected resume
    checkpoint is corrupt, the pipeline should fail fast with a clear error.
    """
    runs = tmp_path / "runs" / "train" / "exp_bad"
    (runs / "weights").mkdir(parents=True, exist_ok=True)
    bad_ckpt = runs / "weights" / "last.pt"
    bad_ckpt.write_bytes(b"CORRUPTED")  # intentionally not a real model file

    def fake_impl_train(**kw):
        resume_from = kw.get("resume_from")
        print(f"[train-abnormal] resume_from={resume_from}")
        # Simulate loader detecting corruption
        if resume_from and Path(resume_from).read_bytes().startswith(b"CORRUPTED"):
            raise RuntimeError("corrupt or mismatched checkpoint detected")
        return 0

    monkeypatch.setattr("user_system.main._impl_train", fake_impl_train, raising=False)

    args = _args([
        "train", "--data", ".", "--names", "cattle,sheep", "--epochs", "1",
        "--resume", "--output", str(runs), "--device", "cpu"
    ])
    with pytest.raises(RuntimeError, match="corrupt or mismatched checkpoint"):
        run_train(args)

def test_edge_missing_optional_results_png(tmp_path, capsys, monkeypatch, install_fake_dataset):
    """
    Edge — missing optional assets: pipeline should succeed even if 'results.png'
    is not produced, as long as core artifacts exist.
    """
    runs = tmp_path / "runs"
    runs.mkdir(parents=True, exist_ok=True)

    def fake_impl_train(**kw):
        run_dir = kw["run_dir"]
        (run_dir / "weights").mkdir(parents=True, exist_ok=True)
        (run_dir / "weights" / "last.pt").write_bytes(b"\x00")
        (run_dir / "args.yaml").write_text("epochs: 1\n")
        (run_dir / "metrics.json").write_text(json.dumps({"epochs": 1}))
        # Intentionally DO NOT write results.png
        (run_dir / "logs.txt").write_text("ok\n")
        (run_dir / "images_vis").mkdir(parents=True, exist_ok=True)
        print(f"[train-edge-missing-optional] run_dir={run_dir}")
        return 0

    monkeypatch.setattr("user_system.main._impl_train", fake_impl_train, raising=False)

    args = _args([
        "train", "--data", ".", "--names", "cattle,sheep", "--epochs", "1",
        "--output", str(runs), "--device", "cpu"
    ])
    run_train(args)

    run_roots = sorted((runs / "train").glob("*"))
    assert run_roots, "No run directory created"
    run_dir = run_roots[-1]

    # Core artifacts must exist
    assert (run_dir / "args.yaml").exists()
    assert (run_dir / "weights" / "last.pt").exists()
    assert (run_dir / "metrics.json").exists()
    # Optional artifact may be missing; ensure test tolerates absence
    assert not (run_dir / "results.png").exists(), "results.png should be intentionally missing"
    # Still, we should have at least two other non-core artifacts
    extra_ok = sum((run_dir / p).exists() for p in ["images_vis", "logs.txt"])
    assert extra_ok >= 1, "Expected at least one optional artifact besides metrics"


import os
import platform

@pytest.mark.skipif(platform.system().lower().startswith("win"), reason="Symlink creation is restricted on Windows")
def test_abnormal_broken_symlink_or_truncated_logs(tmp_path, capsys, monkeypatch, install_fake_dataset):
    """
    Abnormal — broken symlinks / truncated logs:
    - Simulate a run that produces a 'logs.txt' symlink pointing to a missing target,
      or a zero-length (truncated) log if symlinks are not supported.
    - The training entry should not crash on such post-run artifacts; core artifacts
      must still be present.
    """
    runs = tmp_path / "runs"
    runs.mkdir(parents=True, exist_ok=True)

    def fake_impl_train(**kw):
        run_dir = kw["run_dir"]
        (run_dir / "weights").mkdir(parents=True, exist_ok=True)
        (run_dir / "weights" / "last.pt").write_bytes(b"\x00")
        (run_dir / "args.yaml").write_text("epochs: 1\n")
        (run_dir / "metrics.json").write_text(json.dumps({"epochs": 1}))
        (run_dir / "images_vis").mkdir(parents=True, exist_ok=True)

        # Create a broken symlink for logs.txt where supported; otherwise create truncated file
        target = run_dir / "missing_actual_log.txt"
        logs_link = run_dir / "logs.txt"
        try:
            os.symlink(str(target), str(logs_link))
        except (OSError, NotImplementedError):
            # Fallback: create a truncated (zero-length) log file to simulate corruption
            logs_link.write_text("")
        print(f"[train-abnormal-broken-link] run_dir={run_dir}")
        return 0

    monkeypatch.setattr("user_system.main._impl_train", fake_impl_train, raising=False)

    args = _args([
        "train", "--data", ".", "--names", "cattle,sheep", "--epochs", "1",
        "--output", str(runs), "--device", "cpu"
    ])
    run_train(args)

    run_roots = sorted((runs / "train").glob("*"))
    assert run_roots, "No run directory created"
    run_dir = run_roots[-1]

    # Core artifacts must exist regardless of broken/truncated logs
    assert (run_dir / "args.yaml").exists()
    assert (run_dir / "weights" / "last.pt").exists()
    assert (run_dir / "metrics.json").exists()

    # If symlink exists, it should be a link and likely broken (does not exist)
    logs_path = run_dir / "logs.txt"
    if logs_path.exists():
        # Either a truncated regular file or a valid symlink; both are acceptable for this abnormal case
        assert logs_path.is_file() or logs_path.is_symlink()
    else:
        # On some systems, a broken symlink will not count as .exists()
        assert logs_path.is_symlink(), "Expected logs.txt to be a (possibly broken) symlink"
