#!/usr/bin/env python3
"""
---------#
compile.py
- Discovers splits strictly by directory layout:
    images/train <-> labels/train
    images/val or images/valid <-> labels/val|valid
    images/test and test_* variants <-> labels counterparts
    (also supports train/images or images/train variants)

Usage Example:
  python3 compile.py --inputs 'input/*' --out unified \
    --keep-classes "cow,sheep,chicken,pig" \
    --other-class-name "unidentified" \
    --synonyms '{"chook":"chicken","ewe":"sheep"}' \
    --debug
"""
import argparse, hashlib, json, shutil, sys, os, glob
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Iterable
import yaml  # pip install pyyaml

DEBUG = False
def dprint(*args, **kwargs):
    if DEBUG:
        print("[DEBUG]", *args, file=sys.stderr, **kwargs)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp",
            ".JPG", ".JPEG", ".PNG", ".BMP", ".TIF", ".TIFF", ".WEBP"}
CHUNK = 1024 * 1024

def norm(s: str) -> str:
    return " ".join(str(s).strip().lower().split())

def sha256_of_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()

def quick_file_signature(p: Path, head_bytes: int = 64 * 1024) -> tuple:
    """Return a fast, collision-resistant signature used to pre-filter duplicates.
    Includes (size, mtime_ns, sha1(head_bytes)). Full hash only computed on collisions.
    """
    try:
        st = p.stat()
        size = st.st_size
        mtime_ns = st.st_mtime_ns
        h = hashlib.sha1()
        with p.open("rb") as f:
            h.update(f.read(head_bytes))
        return (size, mtime_ns, h.hexdigest())
    except Exception:
        # Fallback to full hash when stat/read fails
        return (p.stat().st_size if p.exists() else -1, 0, sha256_of_file(p))

def parse_yolo_labels_numpy(lbl_path: Path) -> list[str]:
    """Fast YOLO label parsing using numpy; returns list of remappable lines as strings.
    Handles empty or malformed files gracefully by returning an empty list.
    """
    try:
        if not lbl_path.exists() or lbl_path.stat().st_size == 0:
            return []
        # Load as raw strings to avoid conversion errors, then validate tokens per line
        data = lbl_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        return [s.strip() for s in data if s.strip()]
    except Exception:
        return []

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}

def load_yaml_names(yaml_path: Path) -> Dict[int, str]:
    data = load_yaml(yaml_path)
    names = data.get("names")
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {i: str(v) for i, v in enumerate(names)}
    raise ValueError(f"Unsupported 'names' in {yaml_path}")

def find_dataset_yaml(root: Path) -> Optional[Path]:
    for cand in (root/"dataset.yaml", root/"data.yaml", root/"data.yml"):
        if cand.exists():
            return cand
    for ext in ("*.yml", "*.yaml"):
        for y in root.rglob(ext):
            try:
                d = load_yaml(y)
                if isinstance(d, dict) and "names" in d:
                    return y
            except Exception:
                pass
    return None

def expand_inputs(paths: List[Union[Path, str]]) -> List[Path]:
    roots: List[Path] = []
    for p in paths:
        s = str(p).strip().strip('"').strip("'")
        matches = glob.glob(s) or ([s] if Path(s).exists() else [])
        for m in matches:
            q = Path(m).resolve()
            if q.exists():
                roots.append(q if q.is_dir() else q.parent)
    out, seen = [], set()
    for r in roots:
        r = r.resolve()
        if r not in seen:
            seen.add(r); out.append(r)
    return out

def iter_images(dir_: Path) -> Iterable[Path]:
    for p in dir_.rglob("*"):
        if p.is_file() and p.suffix in IMG_EXTS:
            yield p

def label_for(img_path: Path) -> Optional[Path]:
    images_root = None
    for anc in img_path.parents:
        if anc.name.lower() == "images":
            images_root = anc; break
    if images_root is None:
        return None
    rel = img_path.relative_to(images_root)
    labels_root = images_root.parent / "labels"
    cand = (labels_root / rel).with_suffix(".txt")
    return cand if cand.exists() else None

def candidate_split_dirs(base: Path, split: str) -> List[Path]:
    out: List[Path] = []
    alts = ["val","valid"] if split == "val" else [split]

    for s in alts:
        p = base / "images" / s
        if p.exists(): out.append(p)
    for s in alts:
        p = base / s / "images"
        if p.exists(): out.append(p)
    if split == "test":
        base_img = base / "images"
        if base_img.exists():
            out += [q for q in base_img.glob("test_*") if q.is_dir()]
        out += [q/"images" for q in base.glob("test_*") if (q/"images").is_dir()]
    # uniq
    seen, uniq = set(), []
    for d in out:
        d = d.resolve()
        if d not in seen:
            uniq.append(d); seen.add(d)
    return uniq

def discover_pairs_by_dirs(root: Path) -> Dict[str, List[Tuple[Path, Path]]]:
    splits = {"train": [], "val": [], "test": []}
    for split in ("train","val","test"):
        for img_dir in candidate_split_dirs(root, split):
            if img_dir.parent.name.lower() == "images":
                labels_dir = img_dir.parent.parent / "labels" / img_dir.name
            elif img_dir.name.lower() == "images":
                labels_dir = img_dir.parent / "labels"
            else:
                labels_dir = img_dir.parent / "labels" / img_dir.name
            if labels_dir.exists():
                splits[split].append((img_dir.resolve(), labels_dir.resolve()))
    return splits

def main():
    ap = argparse.ArgumentParser(description="Merge YOLO datasets into a single pooled, deduplicated dataset.")
    ap.add_argument("--inputs", nargs="+", required=True, help="Dataset roots or parent folders. Use quotes for globs, e.g., 'input/*'")
    ap.add_argument("--out", required=True, help="Unified output root.")
    ap.add_argument("--keep-classes", default="", help='Comma-separated classes to KEEP (e.g., "cow,sheep,chicken,pig").')
    ap.add_argument("--other-class-name", default="unidentified", help="Name for the collapsed 'other' class.")
    ap.add_argument("--synonyms", default="", help='JSON mapping of alternative names to canonical (e.g., {"ewe":"sheep"}).')
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--debug", action="store_true", help="Verbose debug logging to stderr.")
    
    args = ap.parse_args()
    args.dedup = True
    args.pool_val = True
    args.pool_test = True

    # python dataset/compile.py --inputs ./my_test/inputs/chicken1 ./my_test/inputs/chicken2 ./my_test/inputs/chicken3 ./my_test/inputs/cow1 ./my_test/inputs/cow2 ./my_test/inputs/sheep_3 ./my_test/inputs/uni1 ./my_test/inputs/uni2  --out ./unified
    global DEBUG
    DEBUG = bool(getattr(args, "debug", False))

    inputs = expand_inputs(args.inputs)
    if not inputs:
        raise SystemExit("No dataset roots found under the provided --inputs")
    try:
        synonyms = {norm(k): norm(v) for k, v in (json.loads(args.synonyms) if args.synonyms.strip() else {}).items()}
    except Exception as e:
        print(f"ERROR parsing --synonyms: {e}", file=sys.stderr); sys.exit(2)
    keep = [norm(s) for s in args.keep_classes.split(",") if s.strip()] or ["cow","sheep","chicken","pig"]
    other_name = norm(args.other_class_name)
    canonical = keep + ([other_name] if other_name not in keep else [])
    name_to_gid = {name: i for i, name in enumerate(canonical)}
    gid_other = name_to_gid[other_name] if other_name in name_to_gid else len(canonical)-1

    sources = []
    for r in inputs:
        yml = find_dataset_yaml(r)  # names only
        local_map = {}
        if yml:
            try:
                local_map = load_yaml_names(yml)
            except Exception as e:
                print(f"[WARN] could not read names from {yml}: {e}", file=sys.stderr)
        remap = {}
        for lid, lname in local_map.items():
            ln = synonyms.get(norm(lname), norm(lname))
            remap[lid] = name_to_gid.get(ln, gid_other)
        splits = discover_pairs_by_dirs(r)
        if not any(splits.values()):
            print(f"[WARN] No train/val/test pairs discovered under {r}. Expected images/* and labels/* structure.", file=sys.stderr)
        sources.append({"root": r, "remap": remap, "splits": splits})

    print("=== Plan ===")
    print(f"Unified out: {Path(args.out).resolve()}")
    print(f"Canonical ({len(canonical)}): {canonical}")
    for s in sources:
        print(f"[{s['root']}] remap: {s['remap']}")
        for split, pairs in s["splits"].items():
            for (im,lbl) in pairs:
                print(f"  {split}: images={im}  labels={lbl}")

    if args.dry_run:
        print("\nDry-run complete. No files written.")
        return

    out_root = Path(args.out)
    for split in ("train","val","test"):
        ensure_dir(out_root/"images"/split)
        ensure_dir(out_root/"labels"/split)

    stats = {"processed":0,"added":0,"duplicates":0,"missing_images":0,"missing_labels":0,"malformed_lines":0,"remapped_boxes":0}
    per_class_counts = {split: [0]*len(canonical) for split in ("train","val","test")}
    manifest = {}
    # Two-tier duplicate detection structures
    seen_quick = set()              # set of quick signatures
    seen_hashes = {}                # full-hash -> manifest entry
    lock = threading.Lock()         # protect shared structures when threading
    bad_lines = []

    for src in sources:
        remap = src["remap"]
        for split in ("train","val","test"):
            for (images_dir, labels_dir) in src["splits"][split]:
                tasks = []
                with ThreadPoolExecutor(max_workers=min(32, os.cpu_count() * 4 or 4)) as ex:
                    for img_src in iter_images(images_dir):
                        stats["processed"] += 1

                        lbl = label_for(img_src)
                        if lbl is None:
                            stats["missing_labels"] += 1
                            if DEBUG:
                                dprint("  image has no label:", str(img_src))
                            continue

                        # relative parent structure (keep subfolders by label layout)
                        try:
                            rel = lbl.relative_to(labels_dir)
                            rel_parent = rel.parent
                        except Exception:
                            rel_parent = Path("")

                        if split == "val" and not args.pool_val:
                            dst_labels_base = out_root/"labels"/"val"/src["root"].name
                            dst_images_base = out_root/"images"/"val"/src["root"].name
                        elif split == "test" and not args.pool_test:
                            dst_labels_base = out_root/"labels"/"test"/src["root"].name
                            dst_images_base = out_root/"images"/"test"/src["root"].name
                        else:
                            dst_labels_base = out_root/"labels"/split
                            dst_images_base = out_root/"images"/split

                        def submit_task(img_src=img_src, lbl=lbl, rel_parent=rel_parent, dst_images_base=dst_images_base, dst_labels_base=dst_labels_base):
                            def _work():
                                # Two-tier hashing
                                qsig = quick_file_signature(img_src)
                                with lock:
                                    if args.dedup and qsig in seen_quick:
                                        # possible duplicate, verify with full hash
                                        fh = sha256_of_file(img_src)
                                        if fh in seen_hashes:
                                            return {"dup": 1}
                                    else:
                                        seen_quick.add(qsig)
                                fh = sha256_of_file(img_src)
                                with lock:
                                    if args.dedup and fh in seen_hashes:
                                        return {"dup": 1}
                                hashed_stem = fh[:16]

                                def choose_unique(base_img_dir, base_lbl_dir, parent_rel, stem, img_ext):
                                    idx = 0
                                    while True:
                                        s = stem if idx == 0 else f"{stem}_{idx:02d}"
                                        di = base_img_dir / parent_rel / f"{s}{img_ext.lower()}"
                                        dl = base_lbl_dir / parent_rel / f"{s}.txt"
                                        if args.dedup or (not di.exists() and not dl.exists()):
                                            return di, dl, s
                                        idx += 1

                                dst_image, dst_label, final_stem = choose_unique(
                                    dst_images_base, dst_labels_base, rel_parent, hashed_stem, img_src.suffix
                                )

                                ensure_dir(dst_image.parent)
                                ensure_dir(dst_label.parent)

                                # Parse and remap labels using numpy-backed fast path
                                raw_lines = parse_yolo_labels_numpy(lbl)
                                out_lines = []
                                malformed = 0
                                for ln, s in enumerate(raw_lines, 1):
                                    parts = s.split()
                                    try:
                                        local_id = int(float(parts[0]))
                                    except Exception:
                                        malformed += 1
                                        continue
                                    gid = remap.get(local_id, gid_other)
                                    parts[0] = str(gid)
                                    out_lines.append(" ".join(parts))

                                with dst_label.open("w", encoding="utf-8") as f:
                                    f.write("\n".join(out_lines) + ("\n" if out_lines else ""))

                                shutil.copy2(img_src, dst_image)

                                # prepare result deltas
                                per_class_delta = {}
                                for sline in out_lines:
                                    gid = int(sline.split()[0])
                                    per_class_delta[gid] = per_class_delta.get(gid, 0) + 1

                                with lock:
                                    if args.dedup:
                                        seen_hashes[fh] = {"image": str(dst_image), "label": str(dst_label)}
                                        manifest[fh] = {
                                            "image": str(dst_image.relative_to(out_root).as_posix()),
                                            "label": str(dst_label.relative_to(out_root).as_posix()),
                                        }
                                    else:
                                        k = str(dst_image.relative_to(out_root).as_posix())
                                        manifest[k] = {"image": k, "label": str(dst_label.relative_to(out_root).as_posix())}

                                return {"added": 1, "malformed": malformed, "per_class": per_class_delta}
                            return _work
                        tasks.append(ex.submit(submit_task()))

                    for fut in as_completed(tasks):
                        res = fut.result()
                        if not res:
                            continue
                        if res.get("dup"):
                            stats["duplicates"] += res["dup"]
                            continue
                        stats["added"] += res.get("added", 0)
                        stats["malformed_lines"] += res.get("malformed", 0)
                        # update per-class counts
                        for gid, cnt in (res.get("per_class") or {}).items():
                            per_class_counts[split][gid] += cnt

    dataset_yaml = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(canonical)},
    }
    (out_root/"dataset.yaml").write_text(yaml.safe_dump(dataset_yaml, sort_keys=False), encoding="utf-8")
    (out_root/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    lines = ["split," + ",".join(canonical)]
    for split in ("train","val","test"):
        lines.append(split + "," + ",".join(str(c) for c in per_class_counts[split]))
    (out_root/"per_class_counts.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    if bad_lines:
        (out_root/"bad_lines.log").write_text("\n".join(bad_lines) + "\n", encoding="utf-8")
        stats["malformed_lines"] = len(bad_lines)

    print("\n=== Compile summary ===")
    for k in ("processed","added","duplicates","missing_images","missing_labels","malformed_lines","remapped_boxes"):
        print(f"{k:>16}: {stats[k]}")
    print(f"dataset.yaml  : {out_root/'dataset.yaml'}")
    print(f"dataset root  : {out_root.resolve()}")

if __name__ == "__main__":
    main()
