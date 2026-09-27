# tests/test_train_yolo.py
from pathlib import Path
import pytest
import importlib
import sys, importlib
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
train_yolo = importlib.import_module("train_yolo")

def write_minimal_yolo_dataset(root: Path):
    (root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (root / "images" / "val").mkdir(parents=True, exist_ok=True)
    (root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (root / "labels" / "val").mkdir(parents=True, exist_ok=True)
    (root / "data.yaml").write_text(
        "\n".join([
            f"path: {root}",
            "train: images/train",
            "val: images/val",
            "names: [class0, class1]"
        ])
    )
    return root / "data.yaml"


class FakeResults:
    def __init__(self, d=None):
        self.results_dict = d or {"mAP50-95": 0.123, "precision": 0.5}


class FakeYOLO:
    last_kwargs = None

    def __init__(self, model_name):
        self.model_name = model_name

    def train(self, **kwargs):
        FakeYOLO.last_kwargs = kwargs.copy()
        proj = Path(kwargs.get("project", "."))
        name = kwargs.get("name", "exp")
        workdir = proj / name
        (workdir / "weights").mkdir(parents=True, exist_ok=True)
        (workdir / "weights" / "best.pt").write_bytes(b"FAKE")
        (workdir / "results.csv").write_text("epoch,mAP50-95\n1,0.123\n")
        return FakeResults({"mAP50-95": 0.123, "precision": 0.5})


def test_missing_data_yaml_raises(tmp_path, monkeypatch):
    # Ensure error is raised when data.yaml is missing.
    bad_path = tmp_path / "nope.yaml"
    with pytest.raises(FileNotFoundError):
        train_yolo.train_yolo_api(data=str(bad_path))


def test_detect_yaml_when_given_dir(tmp_path, monkeypatch):
    # Detect data.yaml when a directory is provided and verify artifacts/metrics.
    ds_root = tmp_path / "ds"
    data_yaml = write_minimal_yolo_dataset(ds_root)

    monkeypatch.setattr(train_yolo, "YOLO", FakeYOLO)
    called = {"ok": False}

    def fake_ensure_style_A(p):
        assert Path(p) == data_yaml
        called["ok"] = True

    monkeypatch.setattr(train_yolo, "ensure_style_A", fake_ensure_style_A)

    out = train_yolo.train_yolo_api(
        data=str(ds_root),
        project=str(tmp_path / "runs"),
        name="exp1",
        epochs=2,
        imgsz=320,
        batch=2,
    )

    assert called["ok"] is True
    assert out["workdir"].name == "exp1"
    assert out["best_weights"].name == "best.pt"
    assert out["results_csv"].name == "results.csv"
    assert "mAP50-95" in out["metrics"]


@pytest.mark.parametrize(
    "cuda_ok,mps_ok,expected",
    [
        (True, False, "cuda"),
        (False, True, "mps"),
        (False, False, None),
    ]
)
def test_device_resolution(monkeypatch, tmp_path, cuda_ok, mps_ok, expected):
    # Verify device auto-detection across CUDA/MPS/CPU branches.
    ds_root = tmp_path / "ds"
    write_minimal_yolo_dataset(ds_root)

    def fake_cuda_is_avail():
        return cuda_ok

    class FakeMPS:
        @staticmethod
        def is_available():
            return mps_ok

    monkeypatch.setattr(train_yolo.torch.cuda, "is_available", fake_cuda_is_avail)
    if not hasattr(train_yolo.torch.backends, "mps"):
        train_yolo.torch.backends.mps = FakeMPS()
    else:
        monkeypatch.setattr(train_yolo.torch.backends, "mps", FakeMPS)

    monkeypatch.setattr(train_yolo, "YOLO", FakeYOLO)

    train_yolo.train_yolo_api(
        data=str(ds_root),
        project=str(tmp_path / "runs"),
        name="exp_device"
    )

    dev_sent = FakeYOLO.last_kwargs.get("device", None)
    if expected is None:
        assert dev_sent is None
    else:
        assert dev_sent == expected


def test_hparam_propagation_to_yolo_train(tmp_path, monkeypatch):
    # Check hyperparameters are passed to YOLO.train and artifacts exist.
    ds_root = tmp_path / "ds"
    write_minimal_yolo_dataset(ds_root)

    monkeypatch.setattr(train_yolo, "YOLO", FakeYOLO)

    out = train_yolo.train_yolo_api(
        data=str(ds_root),
        project=str(tmp_path / "runs"),
        name="exp_hparam",
        lr0=0.005,
        lrf=0.1,
        optimizer="adamw",
        cos_lr=True,
        close_mosaic=7,
        epochs=3,
        batch=4,
        imgsz=512
    )

    k = FakeYOLO.last_kwargs
    assert k["lr0"] == 0.005
    assert k["lrf"] == 0.1
    assert k["optimizer"] == "adamw"
    assert k["cos_lr"] is True
    assert k["close_mosaic"] == 7
    assert k["epochs"] == 3
    assert k["batch"] == 4
    assert k["imgsz"] == 512
    assert out["best_weights"].exists()
    assert out["results_csv"].exists()


def test_workdir_and_artifacts(tmp_path, monkeypatch):
    # Validate workdir layout and artifact file paths.
    ds_root = tmp_path / "ds"
    write_minimal_yolo_dataset(ds_root)

    monkeypatch.setattr(train_yolo, "YOLO", FakeYOLO)

    project = tmp_path / "runsX"
    name = "exp_paths"
    out = train_yolo.train_yolo_api(
        data=str(ds_root),
        project=str(project),
        name=name
    )

    assert out["workdir"] == project / name
    assert (project / name / "weights" / "best.pt").exists()
    assert (project / name / "results.csv").exists()


def test_cli_parsing_and_main(monkeypatch, tmp_path, capsys):
    # Run CLI main() with stubbed API and verify args propagation/output lines.
    ds_root = tmp_path / "ds"
    write_minimal_yolo_dataset(ds_root)

    captured = {"args": None}

    def fake_train_yolo_api(**kwargs):
        captured["args"] = kwargs
        workdir = Path(kwargs.get("project") or Path(ds_root).parent / "runs") / kwargs.get("name", "exp")
        (workdir / "weights").mkdir(parents=True, exist_ok=True)
        (workdir / "results.csv").write_text("epoch,mAP50-95\n1,0.123\n")
        return {
            "workdir": workdir,
            "best_weights": workdir / "weights" / "best.pt",
            "results_csv": workdir / "results.csv",
            "metrics": {"mAP50-95": 0.123}
        }

    monkeypatch.setattr(train_yolo, "train_yolo_api", fake_train_yolo_api)

    import sys
    argv_bak = sys.argv
    sys.argv = [
        "train_yolo.py",
        "--data", str(ds_root),
        "--model", "yolov8s.pt",
        "--epochs", "2",
        "--imgsz", "320",
        "--batch", "2",
        "--optimizer", "sgd",
        "--cos-lr",
        "--close-mosaic", "5",
        "--project", str(tmp_path / "runs_cli"),
        "--name", "cli_exp"
    ]

    try:
        train_yolo.main()
    finally:
        sys.argv = argv_bak

    assert captured["args"]["optimizer"] == "sgd"
    assert captured["args"]["cos_lr"] is True
    assert captured["args"]["close_mosaic"] == 5
    assert captured["args"]["epochs"] == 2
    assert captured["args"]["batch"] == 2
    assert captured["args"]["imgsz"] == 320

    out = capsys.readouterr().out
    assert "Training finished" in out
    assert "Workdir" in out
    assert "Best weights" in out
    assert "Results csv" in out


def test_ensure_style_A_called_with_file_path(tmp_path, monkeypatch):
    # Ensure ensure_style_A is called when data is a file path.
    ds_root = tmp_path / "ds"
    data_yaml = write_minimal_yolo_dataset(ds_root)

    monkeypatch.setattr(train_yolo, "YOLO", FakeYOLO)
    flag = {"called": False, "path": None}

    def fake_ensure_style_A(p):
        flag["called"] = True
        flag["path"] = Path(p)

    monkeypatch.setattr(train_yolo, "ensure_style_A", fake_ensure_style_A)

    train_yolo.train_yolo_api(
        data=str(data_yaml),
        project=str(tmp_path / "runs"),
        name="exp_yaml_file"
    )

    assert flag["called"] is True
    assert flag["path"] == data_yaml
