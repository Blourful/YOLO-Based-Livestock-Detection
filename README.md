# YOLO-Based Livestock Detection

A YOLO-based computer vision toolkit for detecting and tracking livestock in images and videos. The project includes dataset preparation, augmentation, training, evaluation, inference, and test utilities.

## Features

- YOLOv8 training with configurable model size, epochs, batch size, image size, optimizer, device, and presets
- Automatic generation and validation of YOLO dataset configuration files
- Image, directory, and video inference
- Object counting and optional tracking support
- Dataset augmentation and class-label management
- Evaluation outputs, metrics, and saved training artifacts
- Unit and integration tests with lightweight test doubles for CI environments

## Project Structure

```text
configs/             Shared training presets
tests/               Project-level tests
train_yolo/          Standalone YOLO training helpers
user_system/         Main training, inference, evaluation, and dataset tools
  configs/           Augmentation and CLI configuration
  dataset/           Dataset compilation utilities
  statistic/         Dataset statistics and plotting
  utils/             Shared helpers
```

## Requirements

- Python 3.10 or newer
- PyTorch
- Ultralytics YOLO
- Optional CUDA-compatible GPU for faster training and inference

Install the pinned dependencies with:

```bash
python -m pip install -r user_system/requirements.txt
```

## Training

Train on an existing YOLO dataset directory or a `data.yaml` file:

```bash
python user_system/train.py \
  --data user_system/tests/cow \
  --size s \
  --epochs 50 \
  --batch 16 \
  --imgsz 640
```

Useful options include `--device cpu`, `--device 0`, `--preset quick`, `--resume`, and `--save_json`.

Training artifacts are saved under the selected output directory. The best checkpoint is normally:

```text
runs/train/<experiment>/weights/best.pt
```

Model weight files are intentionally excluded from version control. Place a trained `.pt` file locally and pass its path with `--weights` or `--model` when required.

## Inference and Evaluation

Evaluate an image, directory, or video with the main CLI:

```bash
python user_system/main.py eval \
  --data path/to/input \
  --model path/to/weights/best.pt \
  --output runs/eval
```

The inference module also supports programmatic use through `user_system.inference.evaluate_model`.

## Dataset Format

The training data should follow a standard YOLO layout. For example:

```text
dataset/
  train/images/
  train/labels/
  valid/images/
  valid/labels/
  test/images/
  test/labels/
  data.yaml
```

The `data.yaml` file defines the image splits and class names.

## Testing

Run the complete test suite with:

```bash
python -m pytest -q
```

Some dataset integrity tests require generated dataset artifacts that are not included in every checkout.
