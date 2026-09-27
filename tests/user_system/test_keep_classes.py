# tests/user_system/test_keep_classes.py
from types import SimpleNamespace
from pathlib import Path
import pytest

import user_system.utils.arg_helper as arg_helper


def test_prepare_dataset_returns_paths(monkeypatch, tmp_path):
    # keep the existing "success" path semantics intact
    srcs = [tmp_path / "A", tmp_path / "B"]
    for s in srcs:
        s.mkdir()
    out = tmp_path / "unified"
    names = ["cattle", "sheep"]

    # prevent real 'dataset/compile.py' call
    def fake_run_compile_subprocess(inputs, out_dir, names=None):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        return (Path(out_dir), {"ok": True})

    monkeypatch.setattr(arg_helper, "run_compile_subprocess", fake_run_compile_subprocess)

    args = SimpleNamespace(compile_inputs=srcs, data=out, names=",".join(names))
    out_path = arg_helper.prepare_dataset(args)
    assert out.exists() and out_path == out


def test_compile_bad_source_path_abnormal(monkeypatch, tmp_path):
    missing = tmp_path / "nope"
    out = tmp_path / "unified"
    args = SimpleNamespace(compile_inputs=[missing], data=out, names="cattle,sheep")

    def fake_run_compile_subprocess(inputs, out_dir, names=None):
        raise FileNotFoundError(str(inputs[0]))

    monkeypatch.setattr(arg_helper, "run_compile_subprocess", fake_run_compile_subprocess)

    # prepare_dataset currently raises SystemExit if inputs missing
    with pytest.raises(SystemExit):
        arg_helper.prepare_dataset(args)


def test_auto_split_edge_reproducible_seed(monkeypatch, tmp_path):
    # pass through compile returning deterministic out_dir
    monkeypatch.setattr(
        arg_helper,
        "reshuffle_or_compile",
        lambda inputs, out_dir, names=None, fallback_data=None, seed=42, **k: (out_dir, {}),
        raising=False,
    )
    srcs = [tmp_path / "A"]
    srcs[0].mkdir()
    out = tmp_path / "unified"
    out_path = arg_helper.prepare_dataset(SimpleNamespace(compile_inputs=srcs, data=out, names="cattle,sheep"))
    assert str(out_path).endswith("unified")


def test_auto_split_abnormal_zero_images(monkeypatch, tmp_path):
    # simulate compilation failure (no images)
    monkeypatch.setattr(
        arg_helper,
        "reshuffle_or_compile",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("no images")),
        raising=False,
    )
    srcs = [tmp_path / "A"]
    srcs[0].mkdir()
    out = tmp_path / "unified"
    with pytest.raises(ValueError):
        arg_helper.prepare_dataset(SimpleNamespace(compile_inputs=srcs, data=out, names="cattle,sheep"))
