# tests/test_statistical_analysis.py
"""
Pytest suite for statistical_analysis.py (Phase-1 dataset analysis).

Covers:
1. End-to-end happy path
2. BBox metrics correctness
3. Drift near zero on identical splits
4. Missing/empty labels integrity
5. Duplicate detection across splits
6. Malformed label lines robustness
7. Extreme geometry aspect ratio propagation
8. Non-RGB image modes (grayscale, RGBA)

Run from project root:
    pytest -v
"""

import sys, subprocess, csv
from pathlib import Path
from PIL import Image, ImageDraw
import pytest

# ----------------- Helpers -----------------

def write_img(path: Path, size=(640,480), color=(180,180,180)):
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.new("RGB", size, color)
    d = ImageDraw.Draw(im)
    w, h = size
    # draw a rectangle for some edges
    d.rectangle([w//4, h//4, 3*w//4, 3*h//4], outline=(20,20,20), width=3)
    im.save(path)

def write_lbl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(" ".join(map(str, r)) + "\n")

def make_yolo_root(tmpdir: Path, name="toyset"):
    root = tmpdir / name
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
    # dataset.yaml
    (root / "dataset.yaml").write_text("names: {0: cow, 1: sheep}\n", encoding="utf-8")
    return root

def run_script(unified_root: Path, out_dir: Path, dataset_name="toyset"):
    cmd = [sys.executable, "statistical_analysis.py",
           "--unified", str(unified_root),
           "--dataset-name", dataset_name,
           "--out-dir", str(out_dir)]
    return subprocess.run(cmd, capture_output=True, text=True)

# ----------------- Tests -----------------

def test_end_to_end_basic(tmp_path):
    root = make_yolo_root(tmp_path, "toyset")

    img_tr = root / "images/train/a.jpg"
    img_va = root / "images/val/b.jpg"
    img_te = root / "images/test/c.jpg"
    write_img(img_tr, (800, 600))
    write_img(img_va, (800, 600))
    write_img(img_te, (1024, 768))

    write_lbl(root / "labels/train/a.txt", [
        [0, 0.5, 0.5, 0.4, 0.3],
        [1, 0.6, 0.6, 0.2, 0.2],
    ])
    write_lbl(root / "labels/val/b.txt", [
        [0, 0.4, 0.5, 0.3, 0.25],
    ])
    write_lbl(root / "labels/test/c.txt", [
        [1, 0.5, 0.5, 0.1, 0.1],
    ])

    out_dir = tmp_path / "reports"; out_dir.mkdir()
    res = run_script(root, out_dir, "toyset")
    assert res.returncode == 0, res.stderr

    class_csv = out_dir / "toyset_class_stats.csv"
    drift_csv = out_dir / "toyset_split_drift.csv"
    assert class_csv.exists()
    assert drift_csv.exists()

    with class_csv.open() as f:
        rows = list(csv.DictReader(f))
    cow_all = next(r for r in rows if r["class_name"]=="cow" and r["split"]=="ALL")
    assert float(cow_all["num_instances"]) >= 1
    assert float(cow_all["med_megapixels"]) > 0
    assert float(cow_all["med_entropy"]) > 0
    assert float(cow_all["med_blur_var"]) > 0

    with drift_csv.open() as f:
        feats = {r["feature"] for r in csv.DictReader(f)}
    assert "megapixels" in feats
    assert "aspect_ratio" in feats

def test_bbox_metrics(tmp_path):
    root = make_yolo_root(tmp_path, "boxes")
    write_img(root / "images/train/x.jpg", (1000, 1000))
    write_lbl(root / "labels/train/x.txt", [
        [0, 0.5, 0.5, 0.5, 0.5],
        [0, 0.2, 0.2, 0.2, 0.4],
    ])
    out_dir = tmp_path / "rep"; out_dir.mkdir()
    run_script(root, out_dir, "boxes")

    with (out_dir / "boxes_class_stats.csv").open() as f:
        rows = list(csv.DictReader(f))
    all0 = next(r for r in rows if r["class_id"]=="0" and r["split"]=="ALL")

    area = float(all0["med_box_area_rel"])
    arat = float(all0["med_box_aratio"])
    density = float(all0["med_box_density_per_mp"])
    fg = float(all0["med_fg_coverage_pct"])

    assert 0.05 <= area <= 0.30
    assert 0.45 <= arat <= 2.1
    assert density > 0
    assert 0.08 <= fg <= 0.33

def test_drift_near_zero_for_identical_splits(tmp_path):
    root = make_yolo_root(tmp_path, "drift0")
    for split in ("train","val"):
        write_img(root / f"images/{split}/a.jpg", (640,480))
        write_lbl(root / f"labels/{split}/a.txt", [[0, 0.5,0.5, 0.2,0.2]])

    out_dir = tmp_path / "rep"; out_dir.mkdir()
    run_script(root, out_dir, "drift0")

    with (out_dir / "drift0_split_drift.csv").open() as f:
        rows = [r for r in csv.DictReader(f) if r["split_a"]=="train" and r["split_b"]=="val"]
    assert rows
    for r in rows:
        assert float(r["js_divergence"]) <= 0.05

def test_missing_empty_labels(tmp_path):
    root = make_yolo_root(tmp_path, "nolabels")
    write_img(root / "images/train/a.jpg", (320,240))
    (root / "labels/train/a.txt").write_text("", encoding="utf-8")

    res = run_script(root, tmp_path / "o", "nolabels")
    assert res.returncode == 0
    assert "missing/empty labels=1" in res.stdout

def test_duplicates_across_splits(tmp_path):
    root = make_yolo_root(tmp_path, "dups")
    img = Image.new("RGB", (200,200), (100,100,100))
    for split in ("train","val"):
        p = root / f"images/{split}/same.jpg"
        p.parent.mkdir(parents=True, exist_ok=True); img.save(p)
        write_lbl(root / f"labels/{split}/same.txt", [[0,0.5,0.5,0.5,0.5]])

    res = run_script(root, tmp_path / "o", "dups")
    assert "cross-split duplicates≈1" in res.stdout

def test_malformed_label_lines(tmp_path):
    root = make_yolo_root(tmp_path, "badlbl")
    write_img(root / "images/train/a.jpg", (400,300))
    (root / "labels/train/a.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n"
        "this is bad\n"
        "1 x y 0.1 0.1\n", encoding="utf-8"
    )
    out_dir = tmp_path / "o"; out_dir.mkdir()
    res = run_script(root, out_dir, "badlbl")
    assert res.returncode == 0
    assert (out_dir / "badlbl_class_stats.csv").exists()

def test_extreme_aspect_ratio(tmp_path):
    root = make_yolo_root(tmp_path, "geom")
    write_img(root / "images/train/wide.jpg", (1600,200))
    write_lbl(root / "labels/train/wide.txt", [[0,0.5,0.5,0.2,0.2]])
    out = tmp_path / "o"; out.mkdir()
    run_script(root, out, "geom")
    with (out / "geom_class_stats.csv").open() as f:
        rows = list(csv.DictReader(f))
    row = next(r for r in rows if r["class_name"]=="cow" and r["split"]=="ALL")
    ar = float(row["med_aspect_ratio"])
    assert ar > 4.0

def test_non_rgb_images(tmp_path):
    root = make_yolo_root(tmp_path, "modes")
    g = Image.new("L", (300,300), 180)
    g.save(root / "images/train/gray.png")
    write_lbl(root / "labels/train/gray.txt", [[0,0.5,0.5,0.2,0.2]])

    rgba = Image.new("RGBA", (256,256), (120,120,120,200))
    rgba.save(root / "images/val/rgba.png")
    write_lbl(root / "labels/val/rgba.txt", [[1,0.5,0.5,0.3,0.3]])

    out_dir = tmp_path / "o"; out_dir.mkdir()
    run_script(root, out_dir, "modes")

    assert (out_dir / "modes_class_stats.csv").exists()
    assert (out_dir / "modes_split_drift.csv").exists()
