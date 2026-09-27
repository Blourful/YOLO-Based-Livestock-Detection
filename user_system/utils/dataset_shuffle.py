"""
utils/dataset_shuffle.py

Per-source reshuffle:
- For each input dataset root (e.g., 'chicken1'), create out/shuffled_<root.name>
  and build ONLY a fresh images/labels split as 8:1:1 (remainder -> train) using
  short, hashed filenames to avoid Windows MAX_PATH issues.

Merge step:
- After reshuffling all sources, invoke dataset/compile.py via subprocess to merge
  the shuffled_* folders into out/unified_from_shuffled. compile.py will handle:
  - global class-name consolidation
  - per-source ID remapping in label files
  - minimal dataset.yaml writing

Public API:
    reshuffle_datasets(roots, out, ratios=(0.8,0.1,0.1), seed=None)
"""

from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Dict
import random, shutil, hashlib, subprocess, sys, yaml, json, os, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue

SUPPORTED_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CHUNK = 1024 * 1024
QUICK_HASH_BYTES = 64 * 1024
MAX_WORKERS_DEFAULT = max(4, (os.cpu_count() or 4) * 4)
MAX_INFLIGHT = 64
SAFE_STEM_MAX = 40

# --------------------------
# Helpers
# --------------------------

def load_aliases_from_yaml(yaml_path: Path) -> Dict[str, str]:
    """
    Load aliases from flag_helper.yaml and convert to synonyms format for compile.py.
    
    Args:
        yaml_path: Path to the flag_helper.yaml file
        
    Returns:
        Dictionary mapping alias terms to canonical names (e.g., {"cattle": "cow", "COW": "cow"})
    """
    if not yaml_path.exists():
        print(f"[WARNING] Aliases file not found: {yaml_path}")
        return {}
    
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        aliases = data.get('aliases', {})
        synonyms = {}
        
        for canonical_name, alias_list in aliases.items():
            if isinstance(alias_list, list):
                for alias in alias_list:
                    if alias:  # Skip empty strings
                        synonyms[alias.lower().strip()] = canonical_name.lower().strip()
        
        if synonyms:
            print(f"🏷️  Loaded {len(synonyms)} aliases from {yaml_path}")
            print(f"   Aliases: {synonyms}")
        else:
            print(f"[INFO] No aliases found in {yaml_path}")
            
        return synonyms
        
    except Exception as e:
        print(f"[WARNING] Could not load aliases from {yaml_path}: {e}")
        return {}

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

class DirCache:
    def __init__(self) -> None:
        self._seen: set[Path] = set()
        self._lock = threading.Lock()
    def ensure(self, p: Path) -> None:
        with self._lock:
            if p in self._seen:
                return
            p.mkdir(parents=True, exist_ok=True)
            self._seen.add(p)

def sha256_of_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()

def quick_file_signature(p: Path) -> tuple:
    """Fast prefilter signature: (size, mtime_ns, sha1 of first QUICK_HASH_BYTES)."""
    try:
        st = p.stat()
        size = st.st_size
        mtime_ns = st.st_mtime_ns
        h = hashlib.sha1()
        with p.open("rb") as f:
            h.update(f.read(QUICK_HASH_BYTES))
        return (size, mtime_ns, h.hexdigest())
    except Exception:
        return (-1, 0, "")

def fast_validate_label(lbl: Path) -> Tuple[bool, int]:
    """Shallow label validation: returns (valid, first_class_id_or_-1)."""
    try:
        if not lbl.exists() or lbl.stat().st_size == 0:
            return True, -1
        # Light parse first non-empty line
        for line in lbl.read_text(encoding="utf-8", errors="ignore").splitlines():
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            _ = int(float(parts[0]))
            return True, _
        return True, -1
    except Exception:
        return False, -1

def link_or_copy(src: Path, dst: Path) -> None:
    """Try hardlink, then symlink, then copy2."""
    try:
        os.link(src, dst)
        return
    except Exception:
        pass
    try:
        # On Windows symlink may require admin; ignore failures
        os.symlink(src, dst)
        return
    except Exception:
        pass
    shutil.copy2(src, dst)

def safe_stem(stem: str) -> str:
    s = (stem or "img")[:SAFE_STEM_MAX].rstrip(" ._-")
    return s or "img"

def unique_name(stem: str, ext: str, h: str) -> str:
    return f"{stem}__{h[:8]}{ext}"

def iter_images(dir_: Path) -> Iterable[Path]:
    """Recursive directory traversal using os.scandir for lower syscall overhead."""
    if not dir_.exists():
        return
    stack = [dir_]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for entry in it:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            p = Path(entry.path)
                            if p.suffix.lower() in SUPPORTED_IMG_EXT:
                                yield p
                    except Exception:
                        continue
        except Exception:
            continue
def candidate_split_image_dirs(base: Path, split: str) -> List[Path]:
    out: List[Path] = []
    alts = ["val", "valid"] if split == "val" else [split]

    # images/{split}
    for s in alts:
        p = base / "images" / s
        if p.exists():
            out.append(p)

    # {split}/images
    for s in alts:
        p = base / s / "images"
        if p.exists():
            out.append(p)

    # test_* variants
    if split == "test":
        base_img = base / "images"
        if base_img.exists():
            for q in base_img.glob("test_*"):
                if q.is_dir():
                    out.append(q)
        for q in base.glob("test_*"):
            r = q / "images"
            if r.is_dir():
                out.append(r)

    # dedupe while preserving order
    seen, uniq = set(), []
    for d in out:
        if d not in seen:
            uniq.append(d)
            seen.add(d)
    return uniq

def label_for(img_path: Path) -> Optional[Path]:
    """Map images/.../<name>.<ext> -> labels/.../<name>.txt if it exists."""
    images_root = None
    for anc in img_path.parents:
        if anc.name == "images":
            images_root = anc
            break
    if images_root is None:
        return None
    rel = img_path.relative_to(images_root)
    labels_root = images_root.parent / "labels"
    cand = (labels_root / rel).with_suffix(".txt")
    return cand if cand.exists() else None

# --------------------------
# Pair collection
# --------------------------

def collect_pairs(root: Path) -> List[Tuple[Path, Path]]:
    pairs: List[Tuple[Path, Path]] = []

    # scan the standard split dirs first
    for split in ("train", "val", "test"):
        for img_dir in candidate_split_image_dirs(root, split):
            for img in iter_images(img_dir):
                lbl = label_for(img)
                if lbl is not None:
                    pairs.append((img, lbl))

    # fallback: flat images/ with no split subdirs
    flat = root / "images"
    if flat.exists() and all(not (flat / s).exists() for s in ("train", "val", "test")):
        for img in iter_images(flat):
            lbl = label_for(img)
            if lbl is not None:
                pairs.append((img, lbl))

    return pairs


# --------------------------
# Copy logic (short names)
# --------------------------

def copy_pair(img: Path, lbl: Path, out_root: Path, split: str, dir_cache: Optional[DirCache] = None) -> Tuple[Path, Path, str]:
    img_dir = out_root / "images" / split
    lbl_dir = out_root / "labels" / split
    if dir_cache:
        dir_cache.ensure(img_dir); dir_cache.ensure(lbl_dir)
    else:
        ensure_dir(img_dir); ensure_dir(lbl_dir)

    h = sha256_of_file(img)
    stem = safe_stem(img.stem)  # keep path short
    out_img = img_dir / unique_name(stem, img.suffix.lower(), h)
    out_lbl = lbl_dir / unique_name(stem, ".txt", h)

    link_or_copy(img, out_img)
    link_or_copy(lbl, out_lbl)
    return out_img, out_lbl, h

# --------------------------
# Split counts (8:1:1, remainder -> train)
# --------------------------

def split_counts(n: int, ratios=(0.8, 0.1, 0.1)) -> Tuple[int, int, int]:
    t = sum(ratios)
    r0, r1 = ratios[0] / t, ratios[1] / t
    n_val = int(n * r1)                 # floor
    n_test = int(n * (1 - r0 - r1))     # floor
    n_train = n - n_val - n_test        # remainder goes to train
    return n_train, n_val, n_test

# --------------------------
# Public API
# --------------------------
def copy_non_data_tree(src: Path, dst: Path) -> None:
    """
    Copy the entire dataset folder tree EXCEPT any images/ or labels/ directories
    (at any depth). This preserves YAMLs, readmes, configs, etc., while avoiding
    Windows MAX_PATH from huge original filenames inside images/labels.
    """
    if dst.exists():
        shutil.rmtree(dst)

    def _ignore(dirpath: str, names: list[str]):
        # ignore any directory named exactly 'images' or 'labels'
        ignored = set()
        for n in names:
            p = Path(dirpath) / n
            if p.is_dir() and n in {"images", "labels"}:
                ignored.add(n)
        return ignored

    shutil.copytree(src, dst, ignore=_ignore)

def reshuffle_datasets(
    roots: List[Path],
    out: Path,
    ratios=(0.8, 0.1, 0.1),
    seed: Optional[int] = None,
    names: Optional[list[str]] = None,
    max_workers: Optional[int] = None,
    inflight: int = MAX_INFLIGHT
) -> Dict[str, Dict[str, int]]:
    """
    For each dataset root:
      - Create out/shuffled_<name> fresh
      - Build images/{train,val,test} and labels/{train,val,test} with ~8:1:1 from pairs
      - After ALL reshuffles, run dataset/compile.py as a subprocess to merge the shuffled_* folders
        into out/unified_from_shuffled (compile.py handles remapping + dataset.yaml)

    Returns a dict keyed by dataset name with per-split counts; also prints compile results.
    """
    if seed is not None:
        random.seed(seed)

    ensure_dir(out)
    summary: Dict[str, Dict[str, int]] = {}
    shuffled_dirs: List[Path] = []
    max_workers = max_workers or min(32, MAX_WORKERS_DEFAULT)

    for root in roots:
        root = Path(root).resolve()
        if not root.exists():
            raise FileNotFoundError(f"Dataset root not found: {root}")

        ds_name = root.name
        dst = out / f"shuffled_{ds_name}"

        # --- [CHANGED] fresh target dir; NO copytree of the whole source ---
        # Start with a clean copy of everything EXCEPT images/ and labels/
        copy_non_data_tree(root, dst)

        # Now create fresh split dirs for our reshuffled data
        for split in ("train", "val", "test"):
            ensure_dir(dst / "images" / split)
            ensure_dir(dst / "labels" / split)

        # Gather & shuffle pairs FROM ORIGINAL ROOT
        pairs = collect_pairs(root)
        random.shuffle(pairs)

        n = len(pairs)
        n_train, n_val, n_test = split_counts(n, ratios)

        counts = {"train": 0, "val": 0, "test": 0, "pairs_total": n}
        # Duplicate prefilter across all splits in this dataset
        seen_quick: set[tuple] = set()
        dir_cache = DirCache()

        # Resume-friendly checkpoints
        ckpt_paths = {
            "train": dst / ".shuffle_ckpt_train.jsonl",
            "val": dst / ".shuffle_ckpt_val.jsonl",
            "test": dst / ".shuffle_ckpt_test.jsonl",
        }
        done_sets: Dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
        for split, ck in ckpt_paths.items():
            if ck.exists():
                try:
                    for line in ck.read_text(encoding="utf-8", errors="ignore").splitlines():
                        if not line.strip():
                            continue
                        rec = json.loads(line)
                        done_sets[split].add(rec.get("hash", ""))
                except Exception:
                    pass

        def write_ckpt(split: str, h: str, img_out: str, lbl_out: str) -> None:
            rec = {"hash": h, "image": img_out, "label": lbl_out, "ts": time.time()}
            try:
                with ckpt_paths[split].open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
            except Exception:
                pass

        # Build a plan assigning each pair to a split index
        plan: List[Tuple[str, Path, Path]] = []
        for i, (img, lbl) in enumerate(pairs):
            split = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
            plan.append((split, img, lbl))

        # Producer-consumer with bounded inflight
        q: queue.Queue = queue.Queue(maxsize=inflight)
        results_lock = threading.Lock()

        def worker():
            while True:
                item = q.get()
                is_sentinel = item is None
                try:
                    if is_sentinel:
                        # No work for sentinel; just exit after accounting in finally
                        continue
                    split, img, lbl = item
                    # fast integrity/duplicate prefilter
                    valid, _ = fast_validate_label(lbl)
                    if not valid:
                        continue
                    qsig = quick_file_signature(img)
                    dup = False
                    if qsig != (-1, 0, ""):
                        with results_lock:
                            if qsig in seen_quick:
                                dup = True
                            else:
                                seen_quick.add(qsig)
                    if dup:
                        continue

                    # Skip if already in checkpoint
                    fh = sha256_of_file(img)
                    if fh in done_sets[split]:
                        with results_lock:
                            counts[split] += 1
                        continue

                    out_img, out_lbl, h = copy_pair(img, lbl, dst, split, dir_cache)
                    write_ckpt(split, h, str(out_img), str(out_lbl))
                    with results_lock:
                        counts[split] += 1
                finally:
                    q.task_done()
                    if is_sentinel:
                        break

        num_workers = max(2, min(32, max_workers))
        threads = [threading.Thread(target=worker, daemon=True) for _ in range(num_workers)]
        for t in threads: t.start()

        for item in plan:
            q.put(item)

        # signal shutdown
        for _ in threads:
            q.put(None)
        q.join()

        summary[ds_name] = counts
        shuffled_dirs.append(dst)

    # --------------------------
    # Merge using compile.py (subprocess)
    # --------------------------
    compile_py = (Path(__file__).resolve().parents[1] / "dataset" / "compile.py").resolve()
    if not compile_py.exists():
        raise FileNotFoundError(f"compile.py not found at expected path: {compile_py}")

    compiled_out = (out / "unified_from_shuffled").resolve()

    # Build command: python dataset/compile.py --out <compiled_out> --inputs <shuffled_*>
    # Note: we deliberately do NOT pass --names here; compile.py will consolidate/ remap for us.
    cmd = [sys.executable, str(compile_py), "--out", str(compiled_out), "--inputs"]
    cmd += [str(p) for p in shuffled_dirs]
    
    # Load aliases from flag_helper.yaml and add --synonyms flag if found
    flag_helper_path = Path("configs/flag_helper.yaml")
    aliases = load_aliases_from_yaml(flag_helper_path)
    if aliases:
        synonyms_json = json.dumps(aliases)
        cmd += ["--synonyms", synonyms_json]
    
    # Forward keep-classes from flag_helper.yaml if present
    try:
        if flag_helper_path.exists():
            with flag_helper_path.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            keep = cfg.get("keep_classes")
            if isinstance(keep, list) and keep:
                keep_csv = ",".join(str(x) for x in keep)
                cmd += ["--keep-classes", keep_csv]
    except Exception as e:
        print(f"[WARN] Failed to read keep_classes from {flag_helper_path}: {e}")
    
    if names:
        cmd += ["--names"] + [str(p) for p in names]
    print("\n▶ Running compile.py on shuffled folders:")
    print(" ", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)

    if proc.returncode != 0:
        print("\n[compile.py stdout]")
        print(proc.stdout)
        print("\n[compile.py stderr]")
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError("compile.py failed on shuffled folders")

    print("\n✅ compile.py completed on shuffled folders.")
    print(proc.stdout.strip())

    summary["_compiled_out"] = {"path": str(compiled_out)}

    return summary
