#test_remap_from_inputs.py
from pathlib import Path
import json
import re
import pytest

EPS = 1e-9

def load_yaml(path: Path):
    try:
        import yaml
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        data = {}
        m = re.search(r"(?m)^\s*names\s*:\s*(\{.*\}|\[.*\])\s*$", path.read_text("utf-8", errors="ignore"))
        if m:
            try:
                data["names"] = json.loads(m.group(1))
            except Exception:
                pass
        return data

def _norm(s: str) -> str:
    return " ".join(str(s).strip().lower().split())

def _parse_names(obj):
    if obj is None: return {}
    if isinstance(obj, list): return {i: str(v) for i, v in enumerate(obj)}
    if isinstance(obj, dict): return {int(k): str(v) for k, v in obj.items()}
    return {}

def _parse_line(line: str):
    s = line.strip()
    if not s: return None
    parts = s.split()
    try:
        cid = int(float(parts[0]))
        xywh = [float(x) for x in parts[1:5]]
        if len(xywh) != 4: return None
        return cid, xywh
    except Exception:
        return None

def iter_input_datasets(base: Path):
    for p in (base/"input").glob("*"):
        if p.is_dir():
            yield p

def find_dataset_yaml(root: Path):
    for fn in ("dataset.yaml","data.yaml","data.yml","dataset.yml"):
        p = root / fn
        if p.exists():
            return p
    return None

def labels_roots(root: Path):
    out = []
    for split in ("train","val","valid","test"):
        cands = [
            root / split / "labels",
            root / "labels" / split,
        ]
        for c in cands:
            if c.is_dir():
                out.append((("val" if split=="valid" else split), c))
    if not out:
        for c in root.rglob("labels"):
            if c.is_dir():
                parts = [p.name.lower() for p in c.parents]
                split = "train"
                for s in ("train","val","valid","test"):
                    if s in parts:
                        split = "val" if s=="valid" else s
                        break
                out.append((split, c))
    seen = set(); uniq = []
    for s, c in out:
        key = (s, c.resolve())
        if key not in seen:
            seen.add(key); uniq.append((s, c))
    return uniq

def canon_maps(unified_root: Path):
    y = unified_root / "dataset.yaml"
    assert y.exists(), f"missing {y}"
    data = load_yaml(y)
    names = _parse_names(data.get("names"))
    if isinstance(names, dict):
        id2name = {int(k): v for k, v in names.items()}
    else:
        id2name = names
    name2id = {v: k for k, v in id2name.items()}
    return id2name, name2id

def expected_gid(local_id, local_id2name, synonyms, name2id, other_name="unidentified"):
    nm = local_id2name.get(local_id)
    if nm is None:
        return name2id.get(other_name)
    canon = synonyms.get(_norm(nm), _norm(nm))
    return name2id.get(canon, name2id.get(other_name))

from collections import defaultdict

def label_pairs(in_root: Path, unified_root: Path):

    unified_labels = unified_root / "labels"
    split_map = {"train": "train", "val": "val", "valid": "val", "test": "test"}

    uni_index = {s: defaultdict(list) for s in ("train", "val", "test")}
    for s in ("train", "val", "test"):
        udir = unified_labels / s
        if udir.is_dir():
            for p in udir.rglob("*.txt"):
                uni_index[s][p.stem].append(p)

    for split, src_dir in labels_roots(in_root):
        u_split = split_map.get(split, split) 
        for src_lbl in src_dir.rglob("*.txt"):
            stem = src_lbl.stem

            cand = uni_index.get(u_split, {}).get(stem, [])
            if not cand:
                for s in ("train", "val", "test"):
                    if uni_index[s].get(stem):
                        cand = uni_index[s][stem]
                        u_split = s
                        break

            if not cand:
                continue

            yield src_lbl, cand[0]


@pytest.mark.timeout(120)
def test_remap_from_inputs_matches_unified(tmp_path, request):
    repo_root = Path(__file__).resolve().parents[2] 
    dataset_root = repo_root / "dataset"
    unified = dataset_root / "unified"
    assert unified.is_dir(), f"unified not found at {unified}"

    synonyms = {
        "0": "chicken",
        "chook": "chicken",
        "hen": "chicken",
        "rooster": "chicken",
        "cattle": "cow",
        "ewe": "sheep",
        "ram": "sheep",
    }

    id2name, name2id = canon_maps(unified)
    other_id = name2id.get("unidentified")

    sampled = 0
    mismatches = []

    for in_root in iter_input_datasets(dataset_root):
        yml = find_dataset_yaml(in_root)
        if not yml:
            continue  

        src = load_yaml(yml)
        local_id2name = _parse_names(src.get("names"))

        def remap(local_id: int):
            raw_name = local_id2name.get(local_id)
            if raw_name is None:
                return other_id

            normed_name = _norm(raw_name)

            if normed_name in synonyms:
                canonical_name = synonyms[normed_name]
            else:
                canonical_name = normed_name

            if canonical_name in name2id:
                gid = name2id[canonical_name]
            else:
                gid = other_id

            return gid

        for src_lbl, uni_lbl in label_pairs(in_root, unified):
            src_text = src_lbl.read_text("utf-8", errors="ignore")
            src_lines = []
            for ln in src_text.splitlines():
                if ln.strip():
                    src_lines.append(ln)

            uni_text = uni_lbl.read_text("utf-8", errors="ignore")
            uni_lines = []
            for ln in uni_text.splitlines():
                if ln.strip():
                    uni_lines.append(ln)

            if len(src_lines) != len(uni_lines):
                msg = (
                    f"{src_lbl} → {uni_lbl}: line count "
                    f"{len(src_lines)} != {len(uni_lines)}"
                )
                mismatches.append(msg)
                sampled += 1
                if sampled >= 300:
                    break
                continue

            for idx in range(len(src_lines)):
                sline = src_lines[idx]
                uline = uni_lines[idx]

                parsed_s = _parse_line(sline)
                parsed_u = _parse_line(uline)
                if not parsed_s or not parsed_u:
                    continue  

                s_id, _ = parsed_s
                u_id, _ = parsed_u

                expected = remap(s_id)
                if u_id != expected:
                    left = f"{src_lbl}:{idx + 1} expected gid {expected} ({id2name.get(expected)})"
                    right = f"but unified has {u_id} ({id2name.get(u_id)})"
                    mismatches.append(f"{left} {right}")

            sampled += 1
            if sampled >= 300:
                break

        if sampled >= 300:
            break

    if mismatches:
        head = "Remap mismatches:\n- "
        lines = mismatches[:50]
        body = "\n- ".join(lines)
        tail = ""
        if len(mismatches) > 50:
            tail = "\n... (truncated)"
        message = head + body + tail
        pytest.fail(message)
