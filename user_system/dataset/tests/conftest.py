# conftest.py
import json, re, hashlib
from pathlib import Path
import pytest

IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp",
            ".JPG",".JPEG",".PNG",".BMP",".TIF",".TIFF",".WEBP"}

def pytest_addoption(parser):
    parser.addoption("--dataset-root", default="unified",
                     help="Path to compiled dataset root (default: unified)")
    parser.addoption("--check-hash", action="store_true", default=False,
                     help="Enable duplicate detection across splits via SHA-256")
    parser.addoption("--hash-limit", type=int, default=0,
                     help="Max images to hash per split (0 = no limit)")

@pytest.fixture(scope="session")
def dataset_root(request) -> Path:
    return Path(request.config.getoption("--dataset-root")).resolve()

def yaml_fallback_minimal(text: str, root: Path):
    data = {"path": str(root), "train": "images/train", "val": "images/val", "test": "images/test"}
    for k in ("path","train","val","test"):
        m = re.search(rf"^\s*{k}\s*:\s*(.+)$", text, flags=re.M)
        if m:
            v = m.group(1).strip()
            v = re.sub(r"\s+#.*$", "", v).strip().strip("'").strip('"')
            data[k] = v
    if re.search(r"^\s*names\s*:", text, flags=re.M):
        data["names"] = None
    return data

def load_dataset_yaml(root: Path):
    ypath = root / "dataset.yaml"
    if not ypath.exists():
        pytest.skip(f"dataset.yaml not found at {ypath}")
    try:
        import yaml
        return yaml.safe_load(ypath.read_text(encoding="utf-8")) or {}
    except Exception:
        return yaml_fallback_minimal(ypath.read_text("utf-8"), root)

def norm_paths(base: Path, v):
    if v is None:
        return []
    vals = v if isinstance(v, list) else [v]
    out = []
    basep = Path(v) if isinstance(v, str) and v.startswith("/") else base
    for s in vals:
        out.append((basep / s).resolve())
    return out

def split_dirs_from_yaml(root: Path, data: dict):
    p = data.get("path", ".")
    p = Path(p)

    # Resolve base relative to dataset root if it's not absolute
    base = (root / p).resolve() if not p.is_absolute() else p.resolve()

    return {
        "train": norm_paths(base, data.get("train")),
        "val":   norm_paths(base, data.get("val")),
        "test":  norm_paths(base, data.get("test")),
    }


def labels_dir_for(images_dir: Path) -> Path:
    if images_dir.parent.name.lower() == "images":
        return images_dir.parent.parent / "labels" / images_dir.name
    if images_dir.name.lower() == "images":
        return images_dir.parent / "labels"
    return images_dir.parent / "labels" / images_dir.name

def walk_images(d: Path):
    for p in d.rglob("*"):
        if p.is_file() and p.suffix in IMG_EXTS:
            yield p

def label_for_image(img: Path):
    images_root = None
    for anc in img.parents:
        if anc.name.lower() == "images":
            images_root = anc; break
    if images_root is None:
        return None
    rel = img.relative_to(images_root)
    cand = (images_root.parent / "labels" / rel).with_suffix(".txt")
    return cand if cand.exists() else None

def sha256_file(p: Path, chunk=1024*1024) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(chunk), b""):
            h.update(c)
    return h.hexdigest()

def parse_label_line(line: str):
    s = line.strip()
    if not s:
        return None, []
    parts = s.split()
    try:
        cid = int(float(parts[0]))
    except Exception:
        return None, []
    if len(parts) < 5:
        return None, []
    try:
        xywh = [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])]
    except Exception:
        return None, []
    return cid, xywh

@pytest.fixture(scope="session")
def ds_meta(dataset_root):
    data = load_dataset_yaml(dataset_root)
    splits = split_dirs_from_yaml(dataset_root, data)
    names = data.get("names")
    n_classes = None
    if isinstance(names, dict):
        try:
            n_classes = 1 + max(int(k) for k in names.keys())
        except Exception:
            n_classes = len(names)
    elif isinstance(names, list):
        n_classes = len(names)
    return {
        "root": dataset_root,
        "data": data,
        "splits": splits,     
        "n_classes": n_classes
    }

@pytest.fixture(scope="session")
def hash_opts(request):
    return {
        "check_hash": bool(request.config.getoption("--check-hash")),
        "hash_limit": int(request.config.getoption("--hash-limit")),
    }
