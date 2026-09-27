# tests/user_system/helper.py
from pathlib import Path
import pytest
import user_system.train as train_mod


@pytest.fixture
def install_fake_dataset(tmp_path, monkeypatch):
    """
    Creates an empty, but well-formed YOLO split structure under tmp_path and
    patches train._resolve_splits + train.auto_build_data_yaml so training code
    'sees' this fake dataset and a generated data.auto.yaml.

        tmp_path/
          fake_ds/
            train/images/
            val/images/
            test/images/
          data.auto.yaml
    """
    tr = tmp_path / "fake_ds" / "train" / "images"
    va = tmp_path / "fake_ds" / "val" / "images"
    te = tmp_path / "fake_ds" / "test" / "images"
    for p in (tr, va, te):
        p.mkdir(parents=True, exist_ok=True)

    # Return Path objects, not strings
    monkeypatch.setattr(
        train_mod,
        "_resolve_splits",
        lambda p: ({"train": tr, "val": va, "test": te}, tmp_path / "data.auto.yaml"),
        raising=False,
    )

    # Avoid touching real project root; write a tiny data.auto.yaml
    def _fake_auto_build(data_root: Path, out_path: Path, names: list):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            f"path: {tmp_path}\n"
            f"train: {tr}\n"
            f"val: {va}\n"
            "names: {0: cattle, 1: sheep}\n"
        )
        return out_path

    monkeypatch.setattr(train_mod, "auto_build_data_yaml", _fake_auto_build, raising=False)

    # Let tests use/inspect the paths if they want
    return {"train": tr, "val": va, "test": te}
