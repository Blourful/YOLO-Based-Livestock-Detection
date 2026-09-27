# utils/arg_helper.py
from __future__ import annotations
from pathlib import Path
import sys, json, tempfile, subprocess
import yaml
from typing import Iterable, List, Optional, Tuple, Dict, Any
from .dataset_shuffle import load_aliases_from_yaml

# Reuse your existing helpers
# - parse_names_arg: parses inline JSON/YAML or filepath into a dict/list
# - reshuffle_datasets: does per-source 80/10/10 reshuffle
from .compile_helper import parse_names_arg
from .dataset_shuffle import reshuffle_datasets
def find_dataset_yaml(root: Path):
    """
    Find a dataset config file in the given root.
    Matches data.yaml, dataset.yaml, or any *.yaml/yml.
    Returns Path or None.
    """
    # First check the most common names explicitly
    for name in ["dataset.yaml", "data.yaml", "dataset.yml", "data.yml"]:
        candidate = root / name
        if candidate.exists():
            return candidate
    
    # Fallback: wildcard search for any yaml/yml file
    matches = list(root.glob("*.y*ml"))
    if matches:
        return matches[0]  # or return all if you want a list
    
    return None

def parse_class_names(args_names: Optional[str]) -> Optional[Dict[str, Any]]:
    """Return parsed names (dict/list) or None, with friendly logging."""
    if not args_names:
        return None
    print(f"\n🏷️  Parsing class names: {args_names}")
    try:
        return parse_names_arg(args_names)
    except Exception as e:
        print(f"[WARNING] Could not parse names argument: {e}")
        print("   Continuing with automatic class name consolidation...")
        return None


def ensure_all_exist(paths: Iterable[Path]) -> None:
    """Ensure all input dataset paths exist; raise SystemExit if not."""
    print("\n📁 Validating input datasets...")
    for i, p in enumerate(paths, 1):
        if not p.exists():
            raise SystemExit(f"[ERROR] Input dataset {i} not found: {p}")
        print(f"   ✓ Dataset {i}: {p}")


def run_compile_subprocess(
    inputs: List[Path],
    out_dir: Path,
    names: Optional[Dict[str, Any]] = None,
) -> Tuple[Path, Optional[Dict[str, Any]]]:
    """Invoke dataset/compile.py via subprocess with optional names payload file.
       Returns (out_dir, optional_stats_dict)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    compile_cmd = [sys.executable, "dataset/compile.py", "--out", str(out_dir)]
    compile_cmd += ["--inputs"] + [str(p) for p in inputs]

    # Optionally add --keep-classes and --synonyms from configs/flag_helper.yaml
    try:
        proj_root = Path(__file__).resolve().parents[1]
        flag_file = proj_root / "configs" / "flag_helper.yaml"
        if flag_file.exists():
            cfg = yaml.safe_load(flag_file.read_text(encoding="utf-8")) or {}
            keep = cfg.get("keep_classes")
            if isinstance(keep, list) and keep:
                keep_csv = ",".join(str(x) for x in keep)
                compile_cmd += ["--keep-classes", keep_csv]
                
            # Add synonyms support
            synonyms = load_aliases_from_yaml(flag_file)
            if synonyms:
                synonyms_json = json.dumps(synonyms)
                compile_cmd += ["--synonyms", synonyms_json]
    except Exception as e:
        print(f"[WARN] Failed to read configuration from flag_helper.yaml: {e}")

    # --names: pass via a temp file (works for dict or list forms)
    names_tmp_path = None
    if names is not None:
        payload = {"names": names} if isinstance(names, dict) else names
        tf = tempfile.NamedTemporaryFile("w", delete=False, suffix=".json")
        json.dump(payload, tf)
        tf.flush(); tf.close()
        names_tmp_path = tf.name
        compile_cmd += ["--names", names_tmp_path]

    print("\n▶ Running compile.py as subprocess:\n", " ".join(compile_cmd))
    comp_proc = subprocess.run(compile_cmd, capture_output=True, text=True)

    if names_tmp_path:
        try:
            Path(names_tmp_path).unlink(missing_ok=True)
        except Exception:
            pass

    if comp_proc.returncode != 0:
        print(comp_proc.stdout)
        print(comp_proc.stderr, file=sys.stderr)
        raise RuntimeError("compile.py failed")

    # Try to parse trailing JSON line (optional)
    stats = None
    try:
        last_line = comp_proc.stdout.strip().splitlines()[-1]
        stats = json.loads(last_line)
    except Exception:
        print(comp_proc.stdout)

    print("\n✅ DATASET COMPILATION COMPLETED SUCCESSFULLY!")
    return out_dir, stats


def reshuffle_or_compile(
    inputs: List[Path],
    out_dir: Path,
    names: Optional[Dict[str, Any]],
    dataset_shuffle: bool,
    shuffle_seed: Optional[int] = None,
    fallback_data: Optional[str] = None,
) -> Tuple[Path, Optional[Dict[str, Any]]]:
    """
    If dataset_shuffle True → reshuffle per source into 80/10/10 under out_dir/unified_from_shuffle.
    Else → run compile.py.
    Returns (final_dataset_root_path, stats_or_None).
    """
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if dataset_shuffle:
        reshuffled = out_dir
        print(f"\n🔁 Reshuffling per source into 80/10/10 at: {reshuffled}/unified_from_shuffled")
        stats_rs = reshuffle_datasets(inputs, reshuffled, seed=shuffle_seed)
        final_root = reshuffled / "unified_from_shuffled"
        return final_root, {"reshuffle_stats": stats_rs}

    final_root, stats = run_compile_subprocess(inputs, out_dir if out_dir else Path(fallback_data or "./unified"), names)
    return Path(final_root), stats


def prepare_dataset(args) -> Path:
    """
    Consolidated dataset preparation used by both 'compile' and 'train' modes.
    - Validates inputs (if provided)
    - Parses names
    - Reshuffles or compiles
    - Updates args.data to point at the final dataset root
    Returns final dataset root (Path).
    """
    # Decide the compilation output root
    try:
        temp = getattr(args, "compile_out", None)
    except Exception as e:
        temp = None

    try:
        temp1 = args.data
    except Exception as e:
        temp1 = None
    compile_outprep = temp or temp1
    compile_output = Path(compile_outprep)
    names = parse_class_names(getattr(args, "names", None))

    # If user provided sources to compile
    if getattr(args, "compile_inputs", None):
        print("\n🔄 DATASET COMPILATION PHASE")
        print("Compiling datasets before proceeding...")
        inputs = [Path(p).resolve() for p in args.compile_inputs]
        ensure_all_exist(inputs)

        final_root, stats = reshuffle_or_compile(
            inputs=inputs,
            out_dir=compile_output,
            names=names,
            dataset_shuffle=getattr(args, "dataset_shuffle", False),
            shuffle_seed=getattr(args, "shuffle_seed", None),
            fallback_data=compile_outprep,
        )
        yaml_hint = final_root if getattr(args, "dataset_shuffle", False) else (compile_output / "dataset.yaml")
        print("=" * 50)
        print(f"\n📄 Generated dataset.yaml: {yaml_hint}")
        print(f"📁 Dataset root directory: {final_root}")
        if stats and isinstance(stats, dict):
            try:
                # Pretty print reshuffle stats if present
                rs = stats.get("reshuffle_stats") if isinstance(stats, dict) else None
                if isinstance(rs, dict):
                    print("\n📊 Shuffler Summary")
                    header = f"{'📁 Dataset':<24} {'🏋️ Train':>10} {'🧪 Val':>10} {'🧭 Test':>10} {'Σ Total':>12}"
                    print(header)
                    print("-" * len(header))
                    total_train = total_val = total_test = 0
                    for name, counts in rs.items():
                        if name == "_compiled_out":
                            continue
                        tr = counts.get("train", 0); vl = counts.get("val", 0); ts = counts.get("test", 0)
                        tt = tr + vl + ts
                        total_train += tr; total_val += vl; total_test += ts
                        print(f"{name:<24} {tr:>10} {vl:>10} {ts:>10} {tt:>12}")
                    grand_total = total_train + total_val + total_test
                    print("-" * len(header))
                    print(f"{'🧮 TOTAL':<24} {total_train:>10} {total_val:>10} {total_test:>10} {grand_total:>12}")
                    outp = rs.get("_compiled_out", {}).get("path")
                    if outp:
                        print(f"📦 Output: {outp}")
                else:
                    print("📊 stats:", json.dumps(stats, indent=2))
            except Exception:
                pass
        return final_root

    # No compile-inputs: possibly reshuffle existing dataset
    data_path = Path(compile_outprep)
    if not data_path.exists():
        raise SystemExit(f"[ERROR] Dataset not found: {data_path}")
    print(f"\n📁 USING EXISTING DATASET")

    if getattr(args, "dataset_shuffle", False):
        data_root = data_path.resolve()
        reshuffled = compile_output.resolve()
        print(f"\n🔁 Reshuffling dataset at {data_root} -> {reshuffled}/unified_from_shuffle")
        stats_rs = reshuffle_datasets([data_root], reshuffled, seed=getattr(args, "shuffle_seed", None))
        final_root = reshuffled / "unified_from_shuffled"
        print("=" * 50)
        print(f"\n📄 Generated dataset.yaml: {final_root / 'dataset.yaml'} (if created)")
        print(f"📁 Dataset root directory: {final_root}")
        print("📊 reshuffle_stats:", stats_rs)
        return final_root

    # Use as-is
    return data_path
