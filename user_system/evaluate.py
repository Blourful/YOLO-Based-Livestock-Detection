import torch
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
import time
from tqdm import tqdm

# -------------------------------
# Configuration
# -------------------------------
MODEL_PATH = "./sheep/sheep/runs/exp/weights/best.pt"  # Model weights path
DATA_YAML = "./sheep/sheep/data.yaml"  # YOLO dataset yaml
DEVICE = 0 if torch.cuda.is_available() else "cpu"  # Automatically select GPU/CPU
BATCH_SIZE = 1  # Evaluation batch


# -------------------------------
# Helper functions
# -------------------------------
def count_metrics_per_image(pred_boxes, true_boxes):
    """Calculate counting metrics for a single image"""
    pred_count = len(pred_boxes)
    true_count = len(true_boxes)

    mae = abs(pred_count - true_count)
    rmse = (pred_count - true_count) ** 2
    bias = pred_count - true_count
    within1 = 1 if mae <= 1 else 0
    under = 1 if pred_count < true_count else 0
    over = 1 if pred_count > true_count else 0

    return {
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "within1": within1,
        "undercount": under,
        "overcount": over,
    }


# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":

    model = YOLO(MODEL_PATH)
    print(f"Using device: {DEVICE}")

    # Validation images & labels
    val_img_dir = Path("./sheep/sheep/images/val")
    val_label_dir = Path("./sheep/sheep/labels/val")
    img_files = sorted(
        [
            f
            for f in val_img_dir.iterdir()
            if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ]
    )

    all_count_metrics = []
    all_times = []

    # -------------------------------
    # Per-image evaluation
    # -------------------------------
    for img_path in tqdm(img_files, desc="Evaluating per image"):
        start_time = time.time()
        results = model.predict(str(img_path), device=DEVICE, conf=0.4, verbose=False)
        end_time = time.time()

        all_times.append(end_time - start_time)

        pred_boxes = (
            results[0].boxes.xyxy.cpu().numpy()
            if hasattr(results[0].boxes, "xyxy")
            else []
        )

        # Load ground truth boxes
        label_path = val_label_dir / (img_path.stem + ".txt")
        true_boxes = []
        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls, xc, yc, w, h = map(float, parts)
                        true_boxes.append([cls, xc, yc, w, h])

        metrics = count_metrics_per_image(pred_boxes, true_boxes)
        all_count_metrics.append(metrics)

    # -------------------------------
    # Counting metrics summary
    # -------------------------------
    df_count = pd.DataFrame(all_count_metrics)
    summary_count = {
        "count_mae": df_count["mae"].mean(),
        "count_rmse": np.sqrt(df_count["rmse"].mean()),
        "count_bias": df_count["bias"].mean(),
        "within1_rate": df_count["within1"].mean(),
        "undercount_rate": df_count["undercount"].mean(),
        "overcount_rate": df_count["overcount"].mean(),
    }

    # -------------------------------
    # Detection metrics using model.val
    # -------------------------------
    val_results = model.val(data=DATA_YAML, device=DEVICE, batch=BATCH_SIZE)
    det_metrics = val_results.box if hasattr(val_results, "box") else val_results

    summary_detection = {
        "mAP50": float(det_metrics.map50),
        "mAP50_95": float(det_metrics.map),
        "recall_at_p95": float(det_metrics.mr),
        "tp_class_accuracy": float(np.mean(det_metrics.p)),
    }

    # Per-class AP
    per_class_ap = det_metrics.ap if hasattr(det_metrics, "ap") else det_metrics.all_ap
    per_class_ap = np.array(per_class_ap)

    if per_class_ap.size > 0:
        worst_ap = per_class_ap.min()
        worst_class_idx = per_class_ap.argmin()
    else:
        worst_ap = 0
        worst_class_idx = -1

    # -------------------------------
    # Automatically count support (number of ground truth boxes)
    # -------------------------------
    val_label_dir = Path("./sheep/sheep/labels/val")
    class_counts = {}  # {class_idx: total_gt_boxes}
    for label_file in val_label_dir.glob("*.txt"):
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls = int(float(parts[0]))
                    class_counts[cls] = class_counts.get(cls, 0) + 1

    # Get worst_class_support
    support = class_counts.get(worst_class_idx, 0)

    summary_detection.update(
        {
            "worst_class_ap": float(worst_ap),
            "worst_class_support": support,
            "worst_class_idx": int(worst_class_idx),
        }
    )

    # -------------------------------
    # Speed metrics (calculated manually)
    # -------------------------------
    # avg_inference_time = sum(all_times) / len(all_times)
    # fps_bs1 = 1 / avg_inference_time
    # latency_ms_p50 = np.median(all_times) * 1000  # milliseconds

    # summary_speed = {
    #     "fps_bs1": fps_bs1,
    #     "latency_ms_p50": latency_ms_p50,
    # }

    preprocess_time_ms = val_results.speed["preprocess"]  # ms
    inference_time_ms = val_results.speed["inference"]  # ms
    postprocess_time_ms = val_results.speed["postprocess"]  # ms
    total_time_ms = preprocess_time_ms + inference_time_ms + postprocess_time_ms
    fps_bs1 = 1000 / total_time_ms
    latency_ms_p50 = total_time_ms
    summary_speed = {
        "fps_bs1": fps_bs1,
        "latency_ms_p50": latency_ms_p50,
    }

    # -------------------------------
    # Merge all summaries
    # -------------------------------
    summary = {**summary_count, **summary_detection, **summary_speed}

    # -------------------------------
    # Output CSV
    # -------------------------------
    out_csv = Path("evaluation_results.csv")
    pd.DataFrame([summary]).to_csv(out_csv, index=False)

    print("=== Evaluation Summary ===")
    for k, v in summary.items():
        print(f"{k:20}: {v}")
    print(f"Results saved to {out_csv}")
