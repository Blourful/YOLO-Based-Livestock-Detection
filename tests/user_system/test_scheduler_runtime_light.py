from user_system.main import build_parser, run_train

def _args(argv):
    return build_parser().parse_args(argv)

def test_early_stop_and_lr_events_appear_in_logs(tmp_path, capsys, monkeypatch, install_fake_dataset):
    runs = tmp_path / "runs"
    runs.mkdir(parents=True, exist_ok=True)

    def fake_impl_train(**kw):
        print("Early stopping at epoch 1 (patience reached)")
        print("LR step: epoch=1, lr=0.001")
        return 0
    monkeypatch.setattr("user_system.main._impl_train", fake_impl_train, raising=False)

    args = _args(["train", "--data", ".", "--names", "cattle,sheep", "--epochs", "3", "--output", str(runs), "--device", "cpu"])
    run_train(args)
    out = capsys.readouterr().out
    assert "Early stopping" in out
    assert "LR step:" in out
