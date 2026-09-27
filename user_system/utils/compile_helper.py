import json
from pathlib import Path
from typing import Dict, Optional
import yaml
# ---- helpers ----
def parse_names_block(obj) -> Dict[int,str]:
    out: Dict[int,str] = {}
    if obj is None: return out
    if isinstance(obj, list):
        for i, n in enumerate(obj): out[i] = str(n)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            try: out[int(k)] = str(v)
            except Exception: pass
    return out

#CLI
def parse_names_arg(arg: Optional[str]) -> Optional[Dict[int,str]]:
    if not arg: return None
    p = Path(arg)
    if p.exists() and p.suffix.lower() in {".yml",".yaml",".json"} and yaml is not None:
        data = yaml.safe_load(p.read_text("utf-8")) if p.suffix.lower() in {".yml",".yaml"} else json.loads(p.read_text("utf-8"))
        return parse_names_block(data.get("names") if isinstance(data, dict) else data)
    # try inline JSON like '{"0":"cow","1":"sheep"}'
    try:
        data = json.loads(arg)
        return parse_names_block(data)
    except Exception:
        return None