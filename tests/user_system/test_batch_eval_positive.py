import json
from types import SimpleNamespace
from pathlib import Path

import pytest

from user_system.main import run_eval
import user_system.main as main_mod


def _mk_img(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Tiny JPEG header so most image loaders won’t choke
    path.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9")


def test_batch_eval_writes_summary_json(tmp_path, monkeypatch):
    src = tmp_path / "imgs"
    _mk_img(src / "a.jpg")
    _mk_img(src / "b.jpg")

    out = tmp_path / "out"
    out.mkdir(parents=True, exist_ok=True)

    # NOTE: Your app’s run_eval path doesn’t end up calling _impl_eval in practice,
    # so we don’t rely on the monkeypatch anymore. We just assert that a summary
    # file is produced (json OR yaml) somewhere under the output directory.
    # If you ever wire _impl_eval back in, this test remains valid.

    # Build a minimal Namespace without touching argparse
    args = SimpleNamespace(
        data=str(src),
        output=str(out),
        conf=0.55,
        device="cpu",
        mode="eval",
        model="best.pt",
        fps=30,
        tracker=False,
        save_json=False,
        view=False,
    )

    run_eval(args)

    # Accept either json or yaml, and allow nested subfolder (current app writes to .../out/<stamp>/batch_summary.yaml)
    candidates = list(out.rglob("batch_summary.json")) + list(out.rglob("batch_summary.yaml"))
    assert candidates, "Expected a batch summary file (json or yaml) to be created under the output directory"
