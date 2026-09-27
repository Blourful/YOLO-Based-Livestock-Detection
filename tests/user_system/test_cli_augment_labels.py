import sys, importlib
from pathlib import Path

def _ensure_sys_path_for_main():
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

def run_main_with_args(args):
    _ensure_sys_path_for_main()
    import user_system.main as main_mod
    importlib.reload(main_mod)
    old = sys.argv[:]
    try:
        sys.argv = ["user_system/main.py", *args]
        return main_mod.main()   # returns None in help branch
    finally:
        sys.argv = old

def test_cli_labels_smoke():
    rc = run_main_with_args(["help"])
    assert rc in (0, None)
