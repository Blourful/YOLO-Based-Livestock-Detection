import re
import pytest
from user_system.main import build_parser, run_train

def _args(argv):
    return build_parser().parse_args(argv)

@pytest.mark.parametrize(("size", "expected"), [("s", r"yolov8s\.pt"), ("x", r"yolov8x\.pt")])
def test_size_flag_normal_and_edge_logs_weights(tmp_path, capsys, monkeypatch, install_fake_dataset, size, expected):
    out = tmp_path / "runs"
    out.mkdir(parents=True, exist_ok=True)

    # Stub heavy train to only print selected weights
    def fake_impl_train(**kw):
        chosen = kw.get("weights_name", f"yolov8{size}.pt")
        print(f"[train] loaded weights file: {chosen}")
        return 0
    monkeypatch.setattr("user_system.main._impl_train", fake_impl_train, raising=False)

    args = _args([
        "train", "--data", ".", "--names", "cattle,sheep",
        "--epochs", "1", "--size", size, "--output", str(out), "--device", "cpu",
    ])
    run_train(args)
    out_text = capsys.readouterr().out
    assert re.search(expected, out_text), f"Log should mention {expected}"
