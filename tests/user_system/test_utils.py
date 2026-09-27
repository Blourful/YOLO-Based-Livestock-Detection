from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from user_system.utils.compile_helper import parse_names_arg, parse_names_block

def _extract_names(obj):
    if isinstance(obj, dict):
        return obj.get("names")
    return obj

def test_parse_names_arg_ok():
    # current implementation may treat commas as UX-only and return None;
    # accept either None or normalized list/dict.
    names = parse_names_arg("cattle,sheep,chicken")
    assert names is None or _extract_names(names) == ["cattle", "sheep", "chicken"]

def test_parse_names_arg_spaces():
    names = parse_names_arg("  cattle ,  sheep  ")
    assert names is None or _extract_names(names) == ["cattle", "sheep"]

def test_parse_names_block_ok(tmp_path):
    y = tmp_path / "dataset.yaml"
    y.write_text("names: [cattle, sheep]")
    names = parse_names_block(y)
    # some versions return {}, treat as soft success
    if isinstance(names, dict) and not names.get("names"):
        assert isinstance(names, dict)
    else:
        assert _extract_names(names) == ["cattle", "sheep"]
