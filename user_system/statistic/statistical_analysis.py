#!/usr/bin/env python3
"""
statistical_analysis.py — Phase 1 (model-agnostic) dataset metrics + split drift

Works on a YOLO-style unified dataset:
  <unified>/images/{train,val,test}
  <unified>/labels/{train,val,test>  (YOLO txt with: class cx cy w h in [0,1])

Outputs (named by --dataset-name):
  - {dataset_name}_class_stats.csv      # class-level medians/counts + bbox metrics (ALL + per-split)
  - {dataset_name}_split_drift.csv      # train/val/test divergence per feature

Usage:
  single thread: python3 statistical_analysis.py --unified ../dataset/unified
  multi thread : python3 statistical_analysis.py --unified ../dataset/unified --workers 4
  custom name  : python3 statistical_analysis.py --unified ../dataset/unified --dataset-name name
"""

from __future__ import annotations
import sys, csv, yaml, math, hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict

import numpy as np
from PIL import Image, ImageFilter

# --- CONFIG ---
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "val", "test")

# -------------- IO HELPERS --------------

def load_yaml(path: Path) -> dict:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as e:
        sys.exit(f"Failed to read {path}: {e}")


def list_images(unified_root: Path) -> List[Tuple[str, Path, Path]]:
    """Return list of (split, img_path, lbl_path) tuples."""
    out: List[Tuple[str, Path, Path]] = []
    for split in SPLITS:
        img_dir = unified_root / "images" / split
        lbl_dir = unified_root / "labels" / split
        if not img_dir.exists():
            continue
        for p in img_dir.rglob("*"):
            if p.suffix.lower() in IMG_EXTS:
                rel = p.relative_to(img_dir)
                lbl = (lbl_dir / rel).with_suffix(".txt")
                out.append((split, p, lbl))
    return out

# -------------- LABEL PARSING --------------

def parse_yolo_label_file(lbl_path: Path) -> List[Tuple[int, float, float, float, float]]:
    """
    Return list of (class_id, cx, cy, w, h) in normalized [0,1] coords.
    Empty list if file missing/empty.
    """
    if not lbl_path.exists():
        return []
    out: List[Tuple[int,float,float,float,float]] = []
    for ln in lbl_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = ln.strip().split()
        if len(parts) < 5:
            continue
        try:
            cid = int(float(parts[0]))
            cx, cy, w, h = map(float, parts[1:5])
            out.append((cid, cx, cy, w, h))
        except Exception:
            continue
    return out

# -------------- IMAGE METRICS (model-agnostic) --------------

def _to_rgb_np(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.float32)


def _to_gray_np(img: Image.Image) -> np.ndarray:
    arr = _to_rgb_np(img)
    return 0.299 * arr[...,0] + 0.587 * arr[...,1] + 0.114 * arr[...,2]


def metric_resolution_wh(gray: np.ndarray) -> Tuple[int,int,float]:
    h, w = gray.shape[:2]
    return w, h, (w*h)/1e6


def metric_aspect_ratio(w: int, h: int) -> float:
    return float(w)/float(h) if h > 0 else 0.0


def metric_entropy(gray: np.ndarray) -> float:
    hist, _ = np.histogram(np.clip(gray,0,255).astype(np.uint8), bins=256, range=(0,255))
    p = hist.astype(np.float64)
    s = p.sum()
    if s <= 0: return 0.0
    p /= s
    nz = p > 0
    return float(-(p[nz]*np.log2(p[nz])).sum())


def _laplacian_var(gray: np.ndarray) -> float:
    k = np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=np.float32)
    g = gray.astype(np.float32)
    gpad = np.pad(g, 1, mode="reflect")
    out = (
        k[0,1]*gpad[0:-2,1:-1] +
        k[1,0]*gpad[1:-1,0:-2] +
        (-4.0)*gpad[1:-1,1:-1] +
        k[1,2]*gpad[1:-1,2:] +
        k[2,1]*gpad[2:,1:-1]
    )
    return float(out.var())


def metric_bytes_per_pixel(img_path: Path, w: int, h: int) -> float:
    pix = max(1, w*h)
    try:
        size = img_path.stat().st_size
    except Exception:
        return 0.0
    return float(size)/float(pix)


def metric_compression_ratio(img_path: Path, w: int, h: int, channels: int = 3) -> float:
    """Approx raw bytes per pixel (channels) divided by file size per pixel."""
    bpp = metric_bytes_per_pixel(img_path, w, h)
    if bpp <= 0: return 0.0
    return float(channels)/bpp  # larger => more compression


def metric_noref_snr_db(gray: np.ndarray) -> float:
    pil = Image.fromarray(np.clip(gray,0,255).astype(np.uint8))
    gb = pil.filter(ImageFilter.GaussianBlur(radius=1.0))
    gb_np = np.asarray(gb, dtype=np.float32)
    signal = gray.astype(np.float32)
    noise = signal - gb_np
    rms_sig = float(np.sqrt(np.mean(signal**2)) + 1e-8)
    rms_noi = float(np.sqrt(np.mean(noise**2)) + 1e-8)
    return float(20.0 * math.log10(rms_sig / rms_noi))


def rgb_stats(rgb: np.ndarray) -> Tuple[float,float,float,float,float,float]:
    r,g,b = rgb[...,0], rgb[...,1], rgb[...,2]
    return float(r.mean()), float(g.mean()), float(b.mean()), float(r.std()), float(g.std()), float(b.std())


def brightness_stats(gray: np.ndarray) -> Tuple[float,float,float,float]:
    return float(gray.min()), float(gray.max()), float(gray.mean()), float(gray.std())


@dataclass
class ImageRecord:
    split: str
    path: Path
    per_class_counts: Dict[int, int] = field(default_factory=dict)
    # image metrics
    width: int = 0
    height: int = 0
    megapixels: float = 0.0
    aspect_ratio: float = 0.0
    entropy: float = 0.0
    blur_var: float = 0.0
    bytes_per_pixel: float = 0.0
    compression_ratio: float = 0.0
    snr_db: float = 0.0
    r_mean: float = 0.0
    g_mean: float = 0.0
    b_mean: float = 0.0
    r_std:  float = 0.0
    g_std:  float = 0.0
    b_std:  float = 0.0
    gray_min: float = 0.0
    gray_max: float = 0.0
    gray_mean: float = 0.0
    gray_std: float = 0.0
    # bbox-derived (per image, any class)
    bboxes: List[Tuple[int,float,float]] = field(default_factory=list)  # (class_id, w_norm, h_norm)
    box_density_per_mp: float = 0.0     # total boxes / megapixel
    fg_coverage_pct: float = 0.0        # sum of areas (clipped to 1.0)


@dataclass
class ClassBucket:
    class_id: int
    class_name: str
    images: List[ImageRecord] = field(default_factory=list)

    def _float_list(self, attr: str, only_split: Optional[str] = None) -> List[float]:
        vals = []
        for rec in self.images:
            if only_split and rec.split != only_split:
                continue
            v = getattr(rec, attr, None)
            if v is not None:
                vals.append(float(v))
        return vals

    def _bbox_attr_list(self, fn, only_split: Optional[str] = None) -> List[float]:
        """Collect bbox-level attributes for this class across all images (optionally per split)."""
        out: List[float] = []
        for rec in self.images:
            if only_split and rec.split != only_split:
                continue
            for (cid, w, h) in rec.bboxes:
                if cid == self.class_id:
                    try:
                        out.append(float(fn(w, h)))
                    except Exception:
                        continue
        return out

    def aggregates(self, only_split: Optional[str] = None) -> Dict[str, object]:
        imgs = [rec for rec in self.images if (not only_split or rec.split == only_split)]
        num_instances = sum(rec.per_class_counts.get(self.class_id, 0) for rec in imgs)
        num_images = len(imgs)
        split_counts = Counter(rec.split for rec in imgs)
        boxes_per_image = [rec.per_class_counts.get(self.class_id, 0) for rec in imgs]

        def med(vals, default=0.0):
            arr = np.asarray(vals, dtype=np.float64)
            return float(np.median(arr)) if arr.size else default

        # per-class medians over images containing this class
        med_metrics = {
            "med_width":          med(self._float_list("width", only_split)),
            "med_height":         med(self._float_list("height", only_split)),
            "med_megapixels":     med(self._float_list("megapixels", only_split)),
            "med_aspect_ratio":   med(self._float_list("aspect_ratio", only_split)),
            "med_entropy":        med(self._float_list("entropy", only_split)),
            "med_blur_var":       med(self._float_list("blur_var", only_split)),
            "med_bytes_per_pixel":med(self._float_list("bytes_per_pixel", only_split)),
            "med_compression_ratio": med(self._float_list("compression_ratio", only_split)),
            "med_snr_db":         med(self._float_list("snr_db", only_split)),
            "med_r_mean":         med(self._float_list("r_mean", only_split)),
            "med_g_mean":         med(self._float_list("g_mean", only_split)),
            "med_b_mean":         med(self._float_list("b_mean", only_split)),
            "med_r_std":          med(self._float_list("r_std", only_split)),
            "med_g_std":          med(self._float_list("g_std", only_split)),
            "med_b_std":          med(self._float_list("b_std", only_split)),
            "med_gray_min":       med(self._float_list("gray_min", only_split)),
            "med_gray_max":       med(self._float_list("gray_max", only_split)),
            "med_gray_mean":      med(self._float_list("gray_mean", only_split)),
            "med_gray_std":       med(self._float_list("gray_std", only_split)),
            "med_box_density_per_mp": med(self._float_list("box_density_per_mp", only_split)),
            "med_fg_coverage_pct":    med(self._float_list("fg_coverage_pct", only_split)),
        }

        # bbox-level medians for THIS class
        area_list = self._bbox_attr_list(lambda w,h: max(0.0, min(1.0, w*h)), only_split)  # relative area
        aspect_list = self._bbox_attr_list(lambda w,h: (w/h) if h>0 else 0.0, only_split)
        bbox_medians = {
            "med_box_area_rel":    med(area_list),
            "med_box_aratio":      med(aspect_list),
        }

        return {
            "num_images": num_images,
            "num_instances": num_instances,
            "split_train": split_counts.get("train", 0),
            "split_val":   split_counts.get("val", 0),
            "split_test":  split_counts.get("test", 0),
            "avg_boxes_per_image": (sum(boxes_per_image) / num_images) if num_images else 0.0,
            "min_boxes": min(boxes_per_image) if boxes_per_image else 0,
            "max_boxes": max(boxes_per_image) if boxes_per_image else 0,
            **med_metrics,
            **bbox_medians,
        }

# -------------- CORE PROCESSING --------------

def compute_image_metrics(img_path: Path) -> Tuple[Dict[str,float], Optional[np.ndarray], Optional[np.ndarray]]:
    try:
        im = Image.open(img_path); im.load()
    except Exception:
        return ({
            "width":0,"height":0,"megapixels":0.0,"aspect_ratio":0.0,"entropy":0.0,"blur_var":0.0,
            "bytes_per_pixel":0.0,"compression_ratio":0.0,"snr_db":0.0,
            "r_mean":0.0,"g_mean":0.0,"b_mean":0.0,"r_std":0.0,"g_std":0.0,"b_std":0.0,
            "gray_min":0.0,"gray_max":0.0,"gray_mean":0.0,"gray_std":0.0
        }, None, None)

    rgb = _to_rgb_np(im)
    gray = _to_gray_np(im)
    w,h,mp = metric_resolution_wh(gray)
    bpp = metric_bytes_per_pixel(img_path, w, h)

    rmean,gmean,bmean,rstd,gstd,bstd = rgb_stats(rgb)
    gmin,gmax,gmean_,gstd_ = brightness_stats(gray)

    metrics = {
        "width": w, "height": h, "megapixels": mp, "aspect_ratio": metric_aspect_ratio(w,h),
        "entropy": metric_entropy(gray),
        "blur_var": _laplacian_var(gray),
        "bytes_per_pixel": bpp,
        "compression_ratio": metric_compression_ratio(img_path, w, h, 3),
        "snr_db": metric_noref_snr_db(gray),
        "r_mean": rmean, "g_mean": gmean, "b_mean": bmean,
        "r_std": rstd, "g_std": gstd, "b_std": bstd,
        "gray_min": gmin, "gray_max": gmax, "gray_mean": gmean_, "gray_std": gstd_,
    }
    return metrics, rgb, gray


def process_one_tuple(item: Tuple[str, Path, Path]) -> Optional[ImageRecord]:
    split, img_path, lbl_path = item
    labels = parse_yolo_label_file(lbl_path)
    if not labels:
        metrics, _, _ = compute_image_metrics(img_path)
        return ImageRecord(split=split, path=img_path, per_class_counts={}, **metrics)

    # per-image class counts + bbox lists
    cnt = Counter()
    bbox_list: List[Tuple[int,float,float]] = []
    for cid, cx, cy, w, h in labels:
        cnt[cid] += 1
        # clamp to [0,1]
        w = max(0.0, min(1.0, w)); h = max(0.0, min(1.0, h))
        bbox_list.append((cid, w, h))

    metrics, _, _ = compute_image_metrics(img_path)
    # density and fg coverage
    total_boxes = sum(cnt.values())
    mp = max(1e-9, metrics["megapixels"])
    box_density = float(total_boxes) / mp
    fg_cov = float(sum(min(1.0, w*h) for (_, w, h) in bbox_list))
    metrics.update({
        "box_density_per_mp": box_density,
        "fg_coverage_pct": min(1.0, fg_cov),
    })

    return ImageRecord(
        split=split, path=img_path,
        per_class_counts=dict(cnt),
        bboxes=bbox_list,
        **metrics
    )


class DatasetIndex:
    def __init__(self, unified_root: Path, data_yaml: Path, max_workers: int = 1):
        self.unified_root = unified_root
        self.data_yaml = data_yaml
        self.names = self._load_names()
        self.class_buckets: Dict[int, ClassBucket] = {cid: ClassBucket(cid, name) for cid, name in self.names.items()}
        self.max_workers = max(1, int(max_workers))
        # for split drift
        self.feature_by_split: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        # integrity
        self.missing_or_empty_labels = 0
        self.total_images_seen = 0
        self.duplicates_across_splits = 0
        self._hash_seen: Dict[str, str] = {}  # sha1 -> split

    def _load_names(self) -> Dict[int, str]:
        y = load_yaml(self.data_yaml)
        names = y.get("names")
        if isinstance(names, list):
            return {i: str(n) for i,n in enumerate(names)}
        if isinstance(names, dict):
            return {int(k): str(v) for k,v in names.items()}
        raise ValueError("dataset.yaml must contain 'names' as list or dict")

    def _quick_sha1(self, path: Path, nbytes: int = 1_000_000) -> Optional[str]:
        try:
            h = hashlib.sha1()
            with path.open("rb") as f:
                chunk = f.read(nbytes)
                if not chunk:
                    return None
                h.update(chunk)
            return h.hexdigest()
        except Exception:
            return None

    def _record_split_features(self, rec: ImageRecord):
        # choose compact feature set for drift
        feats = {
            "megapixels": rec.megapixels,
            "aspect_ratio": rec.aspect_ratio,
            "entropy": rec.entropy,
            "blur_var": rec.blur_var,
            "gray_mean": rec.gray_mean,
            "gray_std": rec.gray_std,
            "bytes_per_pixel": rec.bytes_per_pixel,
            "snr_db": rec.snr_db,
            "box_density_per_mp": rec.box_density_per_mp,
            "fg_coverage_pct": rec.fg_coverage_pct,
        }
        for k,v in feats.items():
            self.feature_by_split[k][rec.split].append(float(v))

    def build(self) -> "DatasetIndex":
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None

        pairs = list_images(self.unified_root)
        records: List[ImageRecord] = []
        total = len(pairs)
        if total == 0:
            print("[WARN] No images found under", self.unified_root)
            return self

        def _pbar(iterable, **kwargs):
            return tqdm(iterable, **kwargs) if tqdm else iterable

        if self.max_workers <= 1:
            it = _pbar(pairs, desc="Scanning (single-thread)", total=total, unit="img")
            for item in it:
                split, img_path, lbl_path = item
                self.total_images_seen += 1
                # integrity: missing/empty labels count
                labels_raw = parse_yolo_label_file(lbl_path)
                if len(labels_raw) == 0:
                    self.missing_or_empty_labels += 1
                # duplicates across splits (quick hash of first 1MB)
                sha = self._quick_sha1(img_path)
                if sha:
                    if sha in self._hash_seen and self._hash_seen[sha] != split:
                        self.duplicates_across_splits += 1
                    else:
                        self._hash_seen[sha] = split
                # process
                rec = process_one_tuple(item)
                if rec:
                    records.append(rec)
                    self._record_split_features(rec)
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            meta_pairs = []
            for (split, img_path, lbl_path) in pairs:
                self.total_images_seen += 1
                labels_raw = parse_yolo_label_file(lbl_path)
                if len(labels_raw) == 0:
                    self.missing_or_empty_labels += 1
                sha = self._quick_sha1(img_path)
                if sha:
                    if sha in self._hash_seen and self._hash_seen[sha] != split:
                        self.duplicates_across_splits += 1
                    else:
                        self._hash_seen[sha] = split
                meta_pairs.append((split, img_path, lbl_path))
            with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
                futures = [ex.submit(process_one_tuple, it) for it in meta_pairs]
                for f in _pbar(as_completed(futures), desc=f"Scanning ({self.max_workers} threads)",
                               total=total, unit="img"):
                    rec = f.result()
                    if rec:
                        records.append(rec)
                        self._record_split_features(rec)

        # bucket by class
        for rec in records:
            for cid, cnt in rec.per_class_counts.items():
                if cid in self.class_buckets and cnt > 0:
                    self.class_buckets[cid].images.append(rec)
        return self

    # -------------- EXPORTS --------------

    def export_class_csv(self, out_csv: Path) -> None:
        total_images_all = sum(len(b.images) for b in self.class_buckets.values())
        total_instances_all = sum(b.aggregates()["num_instances"] for b in self.class_buckets.values())
        totals_images_by_split: Dict[str,int] = {s: 0 for s in SPLITS}
        totals_instances_by_split: Dict[str,int] = {s: 0 for s in SPLITS}
        for b in self.class_buckets.values():
            for s in SPLITS:
                ags = b.aggregates(only_split=s)
                totals_images_by_split[s] += ags["num_images"]
                totals_instances_by_split[s] += ags["num_instances"]

        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "class_id","class_name","split",
                "num_images","num_instances",
                "images_pct","instances_pct",
                "avg_boxes_per_image","min_boxes","max_boxes",
                # image-level medians
                "med_width","med_height","med_megapixels","med_aspect_ratio",
                "med_entropy","med_blur_var","med_bytes_per_pixel","med_compression_ratio","med_snr_db",
                "med_r_mean","med_g_mean","med_b_mean","med_r_std","med_g_std","med_b_std",
                "med_gray_min","med_gray_max","med_gray_mean","med_gray_std",
                # bbox-level medians (for this class)
                "med_box_area_rel","med_box_aratio",
                # per-image aggregated objectness
                "med_box_density_per_mp","med_fg_coverage_pct",
            ])

            for cid, bucket in sorted(self.class_buckets.items()):
                # ---- ALL row (with percentages + medians)
                ag_all = bucket.aggregates()
                w.writerow([
                    cid, bucket.class_name, "ALL",
                    ag_all["num_images"], ag_all["num_instances"],
                    (ag_all["num_images"]/total_images_all) if total_images_all else 0.0,
                    (ag_all["num_instances"]/total_instances_all) if total_instances_all else 0.0,
                    ag_all["avg_boxes_per_image"], ag_all["min_boxes"], ag_all["max_boxes"],
                    ag_all["med_width"], ag_all["med_height"], ag_all["med_megapixels"], ag_all["med_aspect_ratio"],
                    ag_all["med_entropy"], ag_all["med_blur_var"], ag_all["med_bytes_per_pixel"], ag_all["med_compression_ratio"], ag_all["med_snr_db"],
                    ag_all["med_r_mean"], ag_all["med_g_mean"], ag_all["med_b_mean"], ag_all["med_r_std"], ag_all["med_g_std"], ag_all["med_b_std"],
                    ag_all["med_gray_min"], ag_all["med_gray_max"], ag_all["med_gray_mean"], ag_all["med_gray_std"],
                    ag_all["med_box_area_rel"], ag_all["med_box_aratio"],
                    ag_all["med_box_density_per_mp"], ag_all["med_fg_coverage_pct"],
                ])

                for split in SPLITS:
                    ag = bucket.aggregates(only_split=split)
                    tot_i = totals_images_by_split[split]
                    tot_n = totals_instances_by_split[split]
                    w.writerow([
                        cid, bucket.class_name, split,
                        ag["num_images"], ag["num_instances"],
                        (ag["num_images"]/tot_i) if tot_i else 0.0,
                        (ag["num_instances"]/tot_n) if tot_n else 0.0,
                        ag["avg_boxes_per_image"], ag["min_boxes"], ag["max_boxes"],
                        ag["med_width"], ag["med_height"], ag["med_megapixels"], ag["med_aspect_ratio"],
                        ag["med_entropy"], ag["med_blur_var"], ag["med_bytes_per_pixel"], ag["med_compression_ratio"], ag["med_snr_db"],
                        ag["med_r_mean"], ag["med_g_mean"], ag["med_b_mean"], ag["med_r_std"], ag["med_g_std"], ag["med_b_std"],
                        ag["med_gray_min"], ag["med_gray_max"], ag["med_gray_mean"], ag["med_gray_std"],
                        ag["med_box_area_rel"], ag["med_box_aratio"],
                        ag["med_box_density_per_mp"], ag["med_fg_coverage_pct"],
                    ])

    @staticmethod
    def _hist_range(vals: List[float]) -> Tuple[float,float]:
        if not vals: return (0.0, 1.0)
        vmin, vmax = float(min(vals)), float(max(vals))
        if vmin == vmax:
            vmax = vmin + 1e-6
        pad = 0.05*(vmax - vmin)
        return (vmin - pad, vmax + pad)

    @staticmethod
    def _pmf(vals: List[float], bins: int, rng: Tuple[float,float]) -> np.ndarray:
        if not vals:
            return np.ones(bins, dtype=np.float64)/bins
        hist, _ = np.histogram(np.asarray(vals, dtype=np.float64), bins=bins, range=rng)
        p = hist.astype(np.float64)
        p += 1e-12  # smoothing
        p /= p.sum()
        return p

    @staticmethod
    def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
        m = 0.5*(p+q)
        def _kl(a,b):
            return float(np.sum(a * (np.log(a) - np.log(b))))
        return 0.5*_kl(p,m) + 0.5*_kl(q,m)

    @staticmethod
    def _wasserstein1_from_hist(p: np.ndarray, q: np.ndarray, rng: Tuple[float,float]) -> float:
        # Convert pmfs to CDFs and integrate |cdf_p - cdf_q| over support
        cdf_p = np.cumsum(p)
        cdf_q = np.cumsum(q)
        dx = (rng[1] - rng[0]) / len(p)
        return float(np.sum(np.abs(cdf_p - cdf_q)) * dx)

    def export_split_drift_csv(self, out_csv: Path, bins: int = 50) -> None:
        features = sorted(self.feature_by_split.keys())
        rows = []
        for feat in features:
            split_map = self.feature_by_split[feat]
            # choose pairwise: train vs val, train vs test, val vs test (if present)
            splits_present = [s for s in SPLITS if s in split_map]
            # common range across splits
            all_vals = []
            for s in splits_present:
                all_vals.extend(split_map[s])
            rng = self._hist_range(all_vals)
            pmf = {s: self._pmf(split_map[s], bins=bins, rng=rng) for s in splits_present}

            def add_pair(a,b):
                if a in pmf and b in pmf:
                    js = self._js_divergence(pmf[a], pmf[b])
                    w1 = self._wasserstein1_from_hist(pmf[a], pmf[b], rng)
                    rows.append([feat, a, b, js, w1])

            add_pair("train","val")
            add_pair("train","test")
            add_pair("val","test")

        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["feature","split_a","split_b","js_divergence","wasserstein1"])
            w.writerows(rows)

# -------------- CLI --------------

def main():
    import argparse
    here = Path(__file__).parent

    ap = argparse.ArgumentParser(description="Model-agnostic statistical analysis for YOLO-style datasets (Phase 1).")
    ap.add_argument("--unified", type=Path, required=True,
                    help="Unified dataset root (contains images/, labels/, dataset.yaml)")
    ap.add_argument("--dataset-name", type=str, default=None,
                    help="Name used to prefix outputs. Default: basename of --unified.")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Directory to write outputs. Default: alongside this script.")
    ap.add_argument("--workers", type=int, default=1,
                    help="Threads for image I/O (>=1).")
    args = ap.parse_args()

    data_yaml = args.unified / "dataset.yaml"
    if not data_yaml.exists():
        # also accept Ultralytics name 'data.yaml' or 'dataset.yaml'
        alt = args.unified / "data.yaml"
        if alt.exists():
            data_yaml = alt
        else:
            sys.exit(f"Missing dataset YAML at {args.unified}/dataset.yaml or data.yaml")

    dataset_name = args.dataset_name or args.unified.resolve().name
    out_dir = args.out_dir or here
    out_dir.mkdir(parents=True, exist_ok=True)

    class_csv = out_dir / f"{dataset_name}_class_stats.csv"
    drift_csv = out_dir / f"{dataset_name}_split_drift.csv"

    print(f"[INFO] unified root : {args.unified.resolve()}")
    print(f"[INFO] using YAML   : {data_yaml.resolve()}")
    print(f"[INFO] outputs      :\n  - {class_csv}\n  - {drift_csv}")
    print(f"[INFO] workers      : {args.workers} (threaded)")

    idx = DatasetIndex(unified_root=args.unified, data_yaml=data_yaml, max_workers=args.workers).build()

    idx.export_class_csv(class_csv)
    idx.export_split_drift_csv(drift_csv)
    # integrity summary
    missing = idx.missing_or_empty_labels
    dupes = idx.duplicates_across_splits
    total = idx.total_images_seen
    print(f"[OK] wrote {class_csv}")
    print(f"[OK] wrote {drift_csv}")
    print(f"[HEALTH] images scanned={total}, missing/empty labels={missing}, cross-split duplicates≈{dupes}")

if __name__ == "__main__":
    main()
