from pathlib import Path
import sys
import shutil
import torch
from ultralytics import YOLO

WEIGHTS = "best.pt"
IMG_DIR = "img"
IMG_SIZE = 896
CONF_THRES = 0.25
IOU_THRES = 0.45

CLEAR_OUTPUT = True

def main():
    root = Path(__file__).resolve().parent
    weights = root / WEIGHTS
    img_dir = root / IMG_DIR

    if not weights.exists():
        print(f"[ERROR] Weights file not found: {weights}")
        sys.exit(1)
    if not img_dir.exists():
        print(f"[ERROR] Image directory not found: {img_dir}")
        sys.exit(1)

    out_project = root
    out_name = "output"

    if CLEAR_OUTPUT:
        try:
            target = (out_project / out_name).resolve()
            if target.name == "output" and target.parent == root:
                if target.exists():
                    shutil.rmtree(target)
                    print(f"[INFO] Cleared output directory: {target}")
            else:
                print("[WARN] Safety check failed. Skipped clearing output directory.")
        except Exception as e:
            print(f"[WARN] Failed to clear output directory: {e}")

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device} (CUDA if available)")

    model = YOLO(str(weights))

    results = model.predict(
        source=str(img_dir),
        imgsz=IMG_SIZE,
        conf=CONF_THRES,
        iou=IOU_THRES,
        max_det=300,
        device=device,
        save=True,
        project=str(out_project),
        name=out_name,
        exist_ok=True,
        verbose=False,
    )

    save_dir = None
    if results:
        try:
            save_dir = results[0].save_dir
        except Exception:
            save_dir = out_project / out_name
    print(f"[DONE] Annotated results saved to: {save_dir}")

if __name__ == "__main__":
    main()
