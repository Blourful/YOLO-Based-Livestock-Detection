# === YOLOv8 Training Command (Bash) ===
# Script   : train_yolo.py (your training entry file)
# Options  :
#   --data   : Path to data.yaml (defines train/val/test + class names)
#   --model  : Base YOLO checkpoint (here: yolov8s.pt, the "small" variant)
#   --epochs : Number of training epochs (100)
#   --imgsz  : Training image size (896x896)
#   --batch  : Batch size per iteration (16)
#   --device : GPU device index (0 means first GPU)
#
# Example run:
"""
python3 -u train_yolo.py \
  --data ./unified/data.yaml \
  --model yolov8s.pt \
  --epochs 100 \
  --imgsz 896 \
  --batch 16 \
  --device 0 \
  --lr0 0.01 \
  --lrf 0.01 \
  --optimizer sgd \
  --cos-lr \
  --close-mosaic 15
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import os
import torch
from convert import ensure_style_A

def train_yolo_api(
    data: str,
    model: str = "yolov8s.pt",
    epochs: int = 100,
    batch: int = 16,
    imgsz: int = 640,
    workers: int = 4,
    patience: int = 50,
    seed: int = 42,
    device: str | None = None,
    project: str | None = None,
    name: str = "exp",
    resume: bool = False,
    lr0: float = 0.01,
    lrf: float = 0.01,
    optimizer: str = "sgd",
    cos_lr: bool = False,
    close_mosaic: int = 10

):
    data_arg = Path(data).resolve()
    yaml_path = data_arg / "data.yaml" if data_arg.is_dir() else data_arg
    if not yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found: {yaml_path}")
    ensure_style_A(yaml_path)

    resolved_device = device
    if resolved_device is None:
        try:
            if torch.cuda.is_available():
                resolved_device = "cuda"
            elif torch.backends.mps.is_available():
                resolved_device = "mps"
        except Exception:
            resolved_device = None

    if project:
        project_dir = Path(project).resolve()
    else:
        project_dir = yaml_path.parent / "runs"
    os.makedirs(project_dir, exist_ok=True)
    workdir = project_dir / name

    print("==> Config (from API)")
    print(f"data.yaml : {yaml_path}")
    print(f"model     : {model}")
    print(f"epochs    : {epochs}")
    print(f"batch     : {batch}")
    print(f"imgsz     : {imgsz}")
    print(f"device    : {resolved_device or 'auto'}")
    print(f"project   : {project_dir}")
    print(f"name      : {name}")
    print(f"resume    : {resume}")

    yolo = YOLO(model)
    results = yolo.train(
        data=str(yaml_path),
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        device=resolved_device if resolved_device else None,
        workers=workers,
        patience=patience,
        seed=seed,
        project=str(project_dir),
        name=name,
        resume=resume,
        save=True,
        lr0=lr0,
        lrf=lrf,
        optimizer=optimizer,   # 'sgd'/'adam'/'adamw'/'nadam'
        cos_lr=cos_lr,
        close_mosaic=close_mosaic

    )

    best = workdir / "weights" / "best.pt"
    results_csv = workdir / "results.csv"

    metrics = {}
    try:
        if hasattr(results, "results_dict"):
            metrics = dict(results.results_dict)
        elif isinstance(results, dict):
            metrics = results
    except Exception:
        metrics = {}

    return {
        "workdir": workdir,
        "best_weights": best if best.exists() else None,
        "results_csv": results_csv if results_csv.exists() else None,
        "metrics": metrics,
    }


def main():
    ap = argparse.ArgumentParser(description="Train Ultralytics YOLO on an existing YOLO-format dataset")
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--model", type=str, default="yolov8s.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--project", type=str, default=None)
    ap.add_argument("--name", type=str, default="exp")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--lr0", type=float, default=0.01)
    ap.add_argument("--lrf", type=float, default=0.01)
    ap.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adam", "adamw", "nadam"])
    ap.add_argument("--cos-lr", dest="cos_lr", action="store_true")
    ap.add_argument("--close-mosaic", type=int, default=10)

    args = ap.parse_args()

    out = train_yolo_api(
        data=args.data,
        model=args.model,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
        patience=args.patience,
        seed=args.seed,
        device=args.device,
        project=args.project,
        name=args.name,
        resume=args.resume,
        lr0=args.lr0,
        lrf=args.lrf,
        optimizer=args.optimizer,
        cos_lr=args.cos_lr,
        close_mosaic=args.close_mosaic

    )

    print("\nTraining finished")
    print("Workdir     :", out["workdir"])
    print("Best weights:", out["best_weights"] or "(weights/best.pt under training directory)")
    print("Results csv :", out["results_csv"] or "(not found)")
    print("Metrics keys:", list(out["metrics"].keys()) or "(inspect results.csv)")
    print("\nExample inference:")
    print(f"yolo predict model={out['best_weights'] or (out['workdir']/'weights'/'best.pt')} source=<your_images_or_dir>")


if __name__ == "__main__":
    main()
