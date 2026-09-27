# test_unified.py
from pathlib import Path
import json
import pytest

from conftest import (
    labels_dir_for, walk_images, label_for_image,
    parse_label_line, sha256_file
)

EPS = 1e-6

def _split_pairs(ds_meta):
    out = {}
    for split, img_dirs in ds_meta["splits"].items():
        pairs = []
        for img_dir in img_dirs:
            lbl_dir = labels_dir_for(img_dir)
            if img_dir.exists() and lbl_dir.exists():
                pairs.append((img_dir, lbl_dir))
        out[split] = pairs
    return out

def test_dataset_yaml_present(ds_meta):
    ypath = ds_meta["root"] / "dataset.yaml"
    assert ypath.exists(), f"dataset.yaml missing at {ypath}"

def test_split_dirs_exist(ds_meta):
    pairs = _split_pairs(ds_meta)
    for split, lst in pairs.items():
        assert lst, f"No (images, labels) dirs found for split '{split}'"
        for img_dir, lbl_dir in lst:
            assert img_dir.is_dir(), f"{split}: images dir missing: {img_dir}"
            assert lbl_dir.is_dir(), f"{split}: labels dir missing: {lbl_dir}"

def test_image_label_matching_and_format(ds_meta):
    n_classes = ds_meta["n_classes"]
    pairs = _split_pairs(ds_meta)
    issues = []
    for split, lst in pairs.items():
        for img_dir, lbl_dir in lst:
            for img in walk_images(img_dir):
                lbl = label_for_image(img)
                if lbl is None:
                    issues.append(f"[{split}] image has no label: {img}")
                    continue
                if not lbl.is_file():
                    issues.append(f"[{split}] label not a file: {lbl}")
                    continue
                for i, line in enumerate(lbl.read_text("utf-8", errors="ignore").splitlines(), 1):
                    cid, xywh = parse_label_line(line)
                    if cid is None:
                        issues.append(f"[{split}] malformed line {lbl}:{i}: {line}")
                        continue
                    if n_classes is not None and not (0 <= cid < n_classes):
                        issues.append(f"[{split}] class id out of range {cid} ({n_classes}) at {lbl}:{i}")
                    if any(not (-EPS <= v <= 1+EPS) for v in xywh):
                        issues.append(f"[{split}] coords out of [0,1] at {lbl}:{i}: {xywh}")

            for lbl in lbl_dir.rglob("*.txt"):
                rel = lbl.relative_to(lbl_dir).with_suffix("")
                found = any((img_dir / rel.with_suffix(ext)).exists() for ext in {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp",
                                                                                  ".JPG",".JPEG",".PNG",".BMP",".TIF",".TIFF",".WEBP"})
                if not found:
                    issues.append(f"[{split}] label has no matching image: {lbl}")

    if issues:
        pytest.fail(";\n".join(issues[:100]) + ("\n... (truncated)" if len(issues) > 100 else ""))

def test_no_duplicates_across_splits_by_hash(ds_meta, hash_opts):
    pairs = _split_pairs(ds_meta)
    seen = {} 
    limit = hash_opts["hash_limit"]
    for split, lst in pairs.items():
        hashed = 0
        for img_dir, _ in lst:
            for img in walk_images(img_dir):
                h = sha256_file(img)
                if h in seen and seen[h][0] != split:
                    pytest.fail(f"Duplicate image across splits: {seen[h][0]} ↔ {split} hash={h} file={img}")
                seen[h] = (split, img)
                hashed += 1
                if limit and hashed >= limit:
                    break
            if limit and hashed >= limit:
                break

def test_per_class_counts_csv_totals(ds_meta):
    root = ds_meta["root"]
    csvp = root / "per_class_counts.csv"
    if not csvp.exists():
        pytest.skip("per_class_counts.csv not present")
    lines = [ln.strip() for ln in csvp.read_text("utf-8").splitlines() if ln.strip()]
    header = lines[0].split(",")[1:]   
    csv_totals = {}
    for row in lines[1:]:
        parts = row.split(",")
        split = parts[0]
        nums = [int(x) for x in parts[1:]]
        csv_totals[split] = sum(nums)

    pairs = _split_pairs(ds_meta)
    parsed_totals = {"train":0,"val":0,"test":0}
    for split, lst in pairs.items():
        for _, lbl_dir in lst:
            for lbl in lbl_dir.rglob("*.txt"):
                for line in lbl.read_text("utf-8", errors="ignore").splitlines():
                    cid, xywh = parse_label_line(line)
                    if cid is not None:
                        parsed_totals[split] += 1

    for split in ("train","val","test"):
        assert csv_totals.get(split, 0) == parsed_totals.get(split, 0), \
            f"per_class_counts total mismatch for {split}: csv={csv_totals.get(split)} parsed={parsed_totals.get(split)}"

def test_manifest_points_to_real_files(ds_meta):
    root = ds_meta["root"]
    manp = root / "manifest.json"
    if not manp.exists():
        pytest.skip("manifest.json not present")
    try:
        man = json.loads(manp.read_text("utf-8"))
    except Exception as e:
        pytest.fail(f"manifest.json parse error: {e}")

    checked = 0
    for rec in man.values():
        ip = root / rec["image"]
        lp = root / rec["label"]
        assert ip.exists(), f"manifest image missing: {ip}"
        assert lp.exists(), f"manifest label missing: {lp}"
        checked += 1
        if checked >= 200:
            break
