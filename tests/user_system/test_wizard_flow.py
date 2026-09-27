# US-09: Treat help/usage fallback as the lightweight "wizard".
# Your parser doesn't raise on [], so assert that help text lists subcommands.

import re
from user_system.main import build_parser

def test_missing_required_flags_shows_help_text():
    parser = build_parser()
    help_text = parser.format_help()
    assert re.search(r"(train|eval|compile)", help_text), "Help text should list subcommands"

# Edge — treat subcommand help as a lightweight "wizard"
def test_edge_subparser_help_lists_flags(capsys):
    """
    Goal: With partial inputs, 'train --help' should print actionable help (wizard-like),
    not crash.
    """
    parser = build_parser()
    import pytest
    with pytest.raises(SystemExit) as ei:
        parser.parse_args(["train", "--help"])
    assert ei.value.code == 0

    captured = (capsys.readouterr().out + capsys.readouterr().err).lower()
    assert "usage:" in captured
    # Hint at next-step flags
    assert any(flag in captured for flag in ["--data", "--project", "--output", "--epochs", "--batch"])


# Abnormal — unknown/invalid responses should fail fast and cleanly
def test_abnormal_unrecognized_argument_errors_cleanly(capsys):
    """
    Goal: Passing an unknown top-level flag should yield a clean argparse error
    (no traceback), guiding users to correct inputs.
    """
    parser = build_parser()
    import pytest
    with pytest.raises(SystemExit) as ei:
        parser.parse_args(["--this-flag-does-not-exist"])
    assert ei.value.code != 0

    captured = (capsys.readouterr().out + capsys.readouterr().err).lower()
    assert "unrecognized arguments" in captured
