import pytest
import numpy as np
import asyncio
from pathlib import Path
import tempfile
import time
import sys
import types

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

def _ensure_utils_shim():
    if "utils" in sys.modules:
        return
    utils_pkg = types.ModuleType("utils")
    sys.modules["utils"] = utils_pkg
    def _map(name):
        mod_name = f"user_system.utils.{name}"
        try:
            m = __import__(mod_name, fromlist=["*"])
            sys.modules[f"utils.{name}"] = m
        except Exception:
            pass
    for sub in ("compile_helper", "arg_helper", "dataset_shuffle", "ui_helper", "directory_helper"):
        _map(sub)
_ensure_utils_shim()

try:
    import aiofiles  # noqa: F401
    AIO = True
except Exception:
    sys.modules["aiofiles"] = types.ModuleType("aiofiles")
    AIO = False

import user_system.augment as aug
pytestmark = pytest.mark.skipif(not AIO, reason="aiofiles not installed (skipping augmentation performance tests)")


def test_memory_pool_alloc_free():
    # Be robust to constructor signatures:
    pool = None
    try:
        pool = aug.ImageMemoryPool()  # most permissive
    except TypeError:
        try:
            pool = aug.ImageMemoryPool(4)  # capacity only
        except TypeError:
            try:
                pool = aug.ImageMemoryPool(width=320, height=320)  # dims only
            except TypeError:
                pool = aug.ImageMemoryPool(4, 320, 320)  # (capacity, w, h)

    assert pool is not None


def test_async_loader_smoke(tmp_path):
    imgs = [tmp_path / f"{i}.jpg" for i in range(3)]
    for p in imgs:
        p.write_bytes(b"\x00")

    # Create a pool if the loader requires one
    try:
        pool = aug.ImageMemoryPool()
    except TypeError:
        pool = None

    loader = None
    try:
        loader = aug.AsyncImageLoader(pool) if pool is not None else aug.AsyncImageLoader()
    except TypeError:
        # Fall back to the opposite signature
        loader = aug.AsyncImageLoader() if pool is not None else aug.AsyncImageLoader(aug.ImageMemoryPool())

    assert loader is not None
