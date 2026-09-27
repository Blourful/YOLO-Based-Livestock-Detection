"""
Train Unit Tests aligned to User Stories (US-2.x)

This module validates training data handling and configuration behaviors in user_system/train.py.

Coverage vs User Stories:
- US-2.2 Data Input: auto-detect/validate split layouts; auto-generate data.auto.yaml
- US-2.3 Class Definition: names mapping correctness in generated YAML
- US-2.4 Hyperparameter Configuration: CLI overrides vs presets (covered via unit path)

Conventions for each test docstring:
- Input: what artifacts/arguments are provided
- Expected: explicit assertions on outputs/side effects
- Targets: functions/branches under test
- Acceptance mapping: which Acceptance Criteria the test supports
"""
from pathlib import Path
import importlib


def test_resolve_splits_images_first(tmp_images_first):
    """
    Input: dataset root with images-first layout (images/{train,val}, labels/{train,val})
    Expected: layout == "images-first" and train path points to images/train
    Targets: train._resolve_splits (images-first branch)
    Acceptance mapping: US-2.2 (auto-detect dataset layout correctly)
    """
    train = importlib.import_module("user_system.train")
    img_dirs, layout = train._resolve_splits(tmp_images_first)
    assert layout == "images-first"
    assert (tmp_images_first / "images/train") == Path(img_dirs["train"])


def test_resolve_splits_split_first(tmp_split_first):
    """
    Input: dataset root with split-first layout (train/images, val/images)
    Expected: layout == "split-first" and train path points to train/images
    Targets: train._resolve_splits (split-first branch)
    Acceptance mapping: US-2.2 (auto-detect dataset layout correctly)
    """
    train = importlib.import_module("user_system.train")
    img_dirs, layout = train._resolve_splits(tmp_split_first)
    assert layout == "split-first"
    assert (tmp_split_first / "train/images") == Path(img_dirs["train"])


# ------------------------ tests (bottom of file) ------------------------

def test_auto_build_data_yaml_writes_names_and_paths(tmp_images_first, tmp_path):
    """
    Input: images-first dataset and names = ["cattle","sheep","chicken"]
    Expected:
      - data.auto.yaml created with correct absolute path
      - contains train/val keys and names mapping {0: cattle, 1: sheep, 2: chicken}
    Targets: train.auto_build_data_yaml
    Acceptance mapping: US-2.2 (auto-generate YAML), US-2.3 (class names mapping)
    """
    train = importlib.import_module("user_system.train")
    out_yaml = tmp_path / "data.auto.yaml"
    names = ["cattle", "sheep", "chicken"]
    p = train.auto_build_data_yaml(tmp_images_first, out_yaml, names)

    assert p.exists()
    import yaml
    y = yaml.safe_load(out_yaml.read_text())
    assert y["path"] == str(tmp_images_first.resolve())
    assert "train" in y and "val" in y
    assert y["names"] == {0: "cattle", 1: "sheep", 2: "chicken"}


# Integration-lite: validate error when --names missing for directory input
import subprocess, sys
from pathlib import Path

def test_missing_names_raises(tmp_images_first, tmp_path):
    """
    Input: invoke train.py with --data <dir> and NO --names
    Expected: non-zero exit, error message mentions names requirement
    Targets: train.train_from_args input validation (directory + missing --names)
    Acceptance mapping: US-2.3 (must specify class names for directory inputs)
    """
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "user_system" / "train.py"
    cmd = [sys.executable, str(script), "--data", str(tmp_images_first)]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(repo_root))

    assert r.returncode != 0, f"unexpected returncode=0, stdout:\n{r.stdout}\nstderr:\n{r.stderr}"

    merged = (r.stdout or "") + (r.stderr or "")
    assert "must specify --names" in merged, f"got:\n{merged}"


def test_preset_overridden_by_cli(monkeypatch, tmp_images_first, tmp_path):
    """
    Input:
      - Forged preset quick: epochs=10,batch=16,imgsz=640
      - CLI provides epochs=1,batch=8,imgsz=320
    Expected: Fake YOLO receives epochs=1,batch=8,imgsz=320 (CLI overrides preset)
    Targets: train.train_from_args preset merge vs CLI flags
    Acceptance mapping: US-2.4/US-2.5 (CLI overrides presets; final params honored)
    """
    import importlib, argparse
    train = importlib.import_module("user_system.train")

    # forging a preset
    def fake_load_presets(name):
        return {"epochs": 10, "batch": 16, "imgsz": 640}
    monkeypatch.setattr(train, "load_presets", fake_load_presets)

    # fake YOLO to avoid real training
    class FakeYOLO:
        def __init__(self, w): pass
        def train(self, **kw):
            assert kw["epochs"] == 1
            assert kw["imgsz"] == 320
            assert kw["batch"] == 8
            return type("R", (), {"results_dict": {}})()
    monkeypatch.setitem(sys.modules, "ultralytics", type("U", (), {})())
    sys.modules["ultralytics"].YOLO = FakeYOLO

    # construct a min data.auto.yaml
    out = tmp_path / "out"; out.mkdir()
    data_yaml = out / "data.auto.yaml"
    train.auto_build_data_yaml(tmp_images_first, data_yaml, ["a", "b"])

    args = argparse.Namespace(
        size="s", weights=None, data=str(data_yaml), names=None,
        output=str(out), project="train", name=None,
        epochs=1, batch=8, imgsz=320, lr=None, optimizer=None,
        device=None, seed=42, preset="quick", resume=False,
        save_json=False, conf=None, iou=None
    )
    args._cli_flags = {"--epochs", "--batch", "--imgsz"}
    train.train_from_args(args)

