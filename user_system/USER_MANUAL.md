# YOLO Livestock Detection System - User Manual

## Table of Contents

1. [System Overview](#system-overview)
2. [Installation & Setup](#installation--setup)
3. [Quick Start Guide](#quick-start-guide)
4. [Command Reference](#command-reference)
5. [Configuration Files](#configuration-files)
6. [Dataset Management](#dataset-management)
7. [Training & Evaluation](#training--evaluation)
8. [Data Augmentation](#data-augmentation)
9. [Inference & Detection](#inference--detection)
10. [Troubleshooting](#troubleshooting)
11. [Examples & Use Cases](#examples--use-cases)

---

## System Overview

The YOLO Livestock Detection System is a comprehensive machine learning pipeline designed for detecting and classifying livestock animals (cattle, sheep, chicken, pig) in images and videos. The system provides a unified command-line interface for dataset management, model training, evaluation, and inference.

### Key Features

- **Multi-dataset compilation** with automatic class mapping
- **Advanced data augmentation** with configurable transformations
- **YOLO model training** with multiple size options (s/m/x)
- **Real-time inference** on images and videos
- **Comprehensive evaluation** with detailed metrics
- **Hardware optimization** with GPU acceleration support

---

## Installation & Setup

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- 8GB+ RAM (16GB+ recommended)

### Installation Steps

1. **Clone and navigate to the project directory (For Windows you will have to use Git Bash):**

   ```bash
   cd user_system
   ```

2. **Make setup script executable and run:**

   ```bash
   chmod u+x ./setup.sh
   ./setup.sh
   ```

3. **Activate virtual environment:**

   ```bash
   # Windows Git Bash
   source .venv/Scripts/activate

   # Linux/Mac
   source .venv/bin/activate
   ```

4. **Verify installation:**
   ```bash
   python main.py help
   ```

### Dependencies

The system automatically installs required packages including:

- `ultralytics` (YOLO models)
- `opencv-contrib-python-headless` (CUDA-enabled OpenCV)
- `torch` and `torchvision` (PyTorch)
- `numpy`, `matplotlib`, `pandas` (Data processing)
- `PyYAML` (Configuration files)

---

## Quick Start Guide

### 1. Basic Training

```bash
# Train with existing dataset
python main.py train --data ./my_dataset --epochs 50
```

### 2. Compile Multiple Datasets

```bash
# Compile and train in one command
python main.py train --compile-inputs ./cow_data ./sheep_data --data ./unified --epochs 100
```

### 3. Data Augmentation

```bash
# Augment dataset with default settings
python main.py augment --inputs ./my_dataset
```

### 4. Inference

```bash
# Run inference on your data
python main.py eval --data ./test_data
```

---

## Command Reference

### Main Commands

#### `python main.py train`

Train a YOLO model on livestock detection data.

**Basic Usage:**

```bash
python main.py train --data ./dataset --epochs 100
```

**Advanced Options:**

```bash
python main.py train \
  --data ./dataset \
  --epochs 100 \
  --batch 16 \
  --size s \
  --imgsz 640 \
  --device 0 \
  --preset balanced
```

**Dataset Compilation Options:**

```bash
python main.py train \
  --compile-inputs ./cow_data ./sheep_data \
  --data ./unified \
  --dataset-shuffle \
  --shuffle-seed 42 \
  --epochs 100
```

**Parameters:**

- `--data`: **REQUIRED** - Path to dataset directory or data.yaml file (unless using --compile-inputs). Specifies the training dataset location.
- `--epochs`: Number of training epochs (default: 50). Controls how many times the model sees the entire dataset during training.
- `--batch`: Batch size (default: 16). Number of images processed together in each training step.
- `--size`: Model size - s/m/x (default: s). Controls model complexity: s=small/fast, m=medium/balanced, x=large/accurate.
- `--imgsz`: Input image size (default: 640). Resolution to resize images to during training (640x640 pixels).
- `--device`: Device to use (0,1,2... for GPU, 'cpu' for CPU). Specifies which hardware to use for training.
- `--preset`: Training preset - quick/balanced/strong. Pre-configured training settings for different use cases.
- `--compile-inputs`: List of datasets to compile before training (requires --data or --compile-out). Combines multiple datasets into one.
- `--dataset-shuffle`: Shuffle datasets into 80/10/10 splits. Automatically splits data into train/validation/test sets.
- `--shuffle-seed`: Seed for deterministic shuffling. Ensures reproducible dataset splits.

**Config Files:**

- **Optional**: `configs/flag_helper.yaml` - Used for class mapping when compiling datasets (see Configuration Files section for details).

#### `python main.py compile`

Compile multiple datasets into a unified dataset.

**Usage:**

```bash
python main.py compile \
  --compile-inputs ./dataset1 ./dataset2 ./dataset3 \
  --compile-out ./unified_dataset \
  --dataset-shuffle \
  --names ./configs/flag_helper.yaml
```

**Parameters:**

- `--compile-inputs`: **REQUIRED** - List of dataset directories to compile. Specifies which datasets to combine into one unified dataset.
- `--compile-out`: Output directory for compiled dataset (required if --data not provided). Where to save the combined dataset.
- `--data`: Path to dataset (used as output if compiling) (required if --compile-out not provided). Alternative output location for compiled dataset.
- `--dataset-shuffle`: Enable dataset shuffling. Randomly splits each source dataset into train/validation/test sets.
- `--names`: Class mapping configuration file. YAML file defining how to standardize class names across datasets.
- `--shuffle-seed`: Enable deterministic shuffling logic to ensures replicable result for testing.
  **Config Files:**
- **Recommended**: `configs/flag_helper.yaml` - Defines class aliases and canonical names for automatic class mapping (see Configuration Files section for details).

#### `python main.py labels`

Modify class labels in existing datasets.

**Non-destructive modification:**

```bash
python main.py labels \
  --data ./dataset \
  --old-label cow \
  --new-label cattle \
  --out ./dataset_modified
```

**In-place modification:**

```bash
python main.py labels \
  --data ./dataset \
  --old-label cow \
  --new-label cattle \
  --in-place
```

**Parameters:**

- `--data`: **REQUIRED** - Path to dataset directory. Location of the dataset containing labels to modify.
- `--old-label`: **REQUIRED** - Existing class name to change. The current class name in the dataset that needs to be updated.
- `--new-label`: **REQUIRED** - New class name. The target class name to replace the old label with.
- `--out`: Output directory (default: {dataset}\_modified). Where to save the modified dataset (non-destructive).
- `--in-place`: Modify dataset in place (destructive). Updates the original dataset without creating a copy.

**Config Files:**

- **Not Required**: This command works directly with dataset files and doesn't require configuration files.

#### `python main.py augment`

Generate augmented versions of datasets.

**Basic usage:**

```bash
python main.py augment --inputs ./dataset
```

**Advanced usage:**

```bash
python main.py augment \
  --inputs ./dataset1 ./dataset2 \
  --config ./configs/augment.yaml \
  --workers 4
```

**Parameters:**

- `--inputs`: **REQUIRED** - List of dataset directories to augment. Specifies which datasets to apply data augmentation to.
- `--config`: Path to augmentation configuration file (default: user_system/configs/augment.yaml). YAML file controlling augmentation settings and parameters.
- `--workers`: Number of parallel workers (default: 0 = auto-detect). Controls parallel processing for faster augmentation.

**Config Files:**

- **Required**: `configs/augment.yaml` - Controls all augmentation settings including enabled transformations, probabilities, and performance options (see Configuration Files section for details).

#### `python main.py eval`

Run inference on images, videos, or directories using trained models.

**Usage:**

```bash
python main.py eval \
  --data ./test_dataset \
  --model yolov8n.pt \
  --fps 15 \
  --tracker bt \
  --conf 0.25 \
  --output ./runs/eval
```

**Parameters:**

- `--data`: **REQUIRED** - Path to inference data (image file, video file, or directory). Location of images/videos to run inference on.
- `--model`: Model file (.pt) (default: ./model/default.pt). Trained model weights to use for inference.
- `--fps`: Target FPS for video processing (default: 15). **[Video only]** Controls how many frames per second are processed (higher = more frames analyzed but slower).
- `--tracker`: Tracking algorithm - bt/bs (default: bt). **[Video only]** Object tracking method for video inference (bt=ByteTrack, bs=BotSort).
- `--output`: Output directory (default: ./runs/eval). Where to save inference results and annotated outputs.
- `--conf`: Confidence threshold (default: 0.25). Minimum confidence score for detections.

**Config Files:**

- **Not Required**: This command works with trained models and data directly without requiring configuration files.

#### `python main.py recommend`

Get hardware configuration recommendations.

**Usage:**

```bash
python main.py recommend
```

**Parameters:**

- No parameters required. Analyzes your system hardware and provides optimal configuration recommendations.

**Config Files:**

- **Not Required**: This command analyzes system hardware automatically without requiring configuration files.

#### `python main.py help`

Display help information and usage examples.

**Usage:**

```bash
python main.py help
```

**Parameters:**

- No parameters required. Shows comprehensive help information and command examples.

**Config Files:**

- **Not Required**: This command displays help information without requiring configuration files.

---

## Configuration Files

The system uses YAML configuration files to control various aspects of dataset processing, augmentation, and class mapping. Understanding and properly configuring these files is essential for optimal performance.

### 1. Augmentation Configuration (`configs/augment.yaml`)

**Required for**: `python main.py augment` command

**Purpose**: Controls all data augmentation settings including enabled transformations, probabilities, performance options, and output settings.

**Location**: `user_system/configs/augment.yaml`

**How to Modify**:

1. **Open the config file**:

   ```bash
   # Edit with your preferred text editor
   nano user_system/configs/augment.yaml
   # or
   code user_system/configs/augment.yaml
   ```

2. **Key Configuration Sections**:

   **Output Settings**:

   ```yaml
   output: aug_output # Output directory name
   in_place: false # false = create new dataset, true = modify original
   seed: 123 # Random seed for reproducibility
   keep_original: true # Include original images in output
   ```

   **Augmentation Count**:

   ```yaml
   counts: 1 # Generate N augmented versions per image
   # OR
   multiplier: 3 # Target dataset size multiplier
   ```

   **Performance Settings**:

   ```yaml
   max_workers: null # null = auto-detect, 1 = sequential, 4+ = parallel
   batch_size: 32 # Images processed per batch
   use_optimizations: true # Enable performance optimizations
   use_cuda: true # Enable GPU acceleration
   ```

   **Augmentation Modes**:

   ```yaml
   modes:
     flip_h: # Horizontal flip
       enabled: true
       prob: 0.5 # 50% chance of applying

     rotate: # Rotation
       enabled: true
       prob: 0.3
       max_deg: 12 # Maximum rotation angle

     hsv: # Color variation
       enabled: true
       h: 0.015 # Hue variation
       s: 0.7 # Saturation variation
       v: 0.4 # Brightness variation
   ```

3. **Common Modifications**:

   **For Faster Processing**:

   ```yaml
   max_workers: 8
   batch_size: 64
   use_optimizations: true
   ```

   **For Higher Quality**:

   ```yaml
   counts: 3
   deterministic: true
   ```

   **For Specific Use Cases**:

   ```yaml
   # Disable certain augmentations
   modes:
     grayscale:
       enabled: false
     blur:
       enabled: false
   ```

### 2. Class Mapping Configuration (`configs/flag_helper.yaml`)

**Required for**: `python main.py compile` command (when using class mapping)

**Purpose**: Defines class aliases and canonical names for automatic standardization across different datasets.

**Location**: `user_system/configs/flag_helper.yaml`

**How to Modify**:

1. **Open the config file**:

   ```bash
   nano user_system/configs/flag_helper.yaml
   ```

2. **Configuration Structure**:

   **Class Aliases**:

   ```yaml
   aliases:
     cow: # Canonical name
       - cattle # Alternative names
       - COW
       - bovine
     chicken:
       - Chicken
       - "0" # Numeric class ID
       - hen
       - rooster
     sheep:
       - ewe
       - ram
       - lamb
   ```

   **Keep Classes** (canonical class list):

   ```yaml
   keep_classes:
     - sheep
     - cow
     - chicken
     - pig
     - aerial_sheep
     - aerial_cow
   ```

3. **Adding New Classes**:

   **Step 1**: Add to `keep_classes`:

   ```yaml
   keep_classes:
     - sheep
     - cow
     - chicken
     - pig
     - goat # New class
     - horse # Another new class
   ```

   **Step 2**: Add aliases if needed:

   ```yaml
   aliases:
     goat:
       - kid
       - nanny
     horse:
       - pony
       - stallion
   ```

4. **Common Use Cases**:

   **Standardizing Different Naming Conventions**:

   ```yaml
   aliases:
     cow:
       - cattle
       - COW
       - bovine
       - bull
       - calf
   ```

   **Handling Numeric Class IDs**:

   ```yaml
   aliases:
     chicken:
       - "0" # Class ID 0 maps to chicken
       - "1" # Class ID 1 also maps to chicken
   ```

### 3. Dataset Configuration (`data.yaml`)

**Required for**: All commands that work with datasets

**Purpose**: Defines dataset structure, class names, and paths for YOLO format datasets.

**Location**: Inside each dataset directory (e.g., `./my_dataset/data.yaml`)

**How to Create/Modify**:

1. **Basic Structure**:

   ```yaml
   path: /absolute/path/to/dataset
   train: images/train
   val: images/val
   test: images/test

   nc: 4 # Number of classes
   names: ["cow", "sheep", "chicken", "pig"]
   ```

2. **Relative Paths** (recommended):

   ```yaml
   path: . # Current directory
   train: images/train
   val: images/val
   test: images/test

   nc: 4
   names: ["cow", "sheep", "chicken", "pig"]
   ```

3. **Adding New Classes**:
   ```yaml
   nc: 6 # Update count
   names: ["cow", "sheep", "chicken", "pig", "goat", "horse"]
   ```

### Configuration File Usage Examples

**Example 1: Custom Augmentation for Small Datasets**

```yaml
# configs/augment.yaml
counts: 5 # Generate 5x more data
deterministic: true # Apply all enabled augmentations
modes:
  flip_h:
    enabled: true
    prob: 0.5
  rotate:
    enabled: true
    prob: 0.3
    max_deg: 15
  hsv:
    enabled: true
    h: 0.02
    s: 0.8
    v: 0.5
```

**Example 2: Class Mapping for Mixed Datasets**

```yaml
# configs/flag_helper.yaml
aliases:
  cow:
    - cattle
    - COW
    - bovine
    - "0" # Some datasets use numeric IDs
  sheep:
    - ewe
    - ram
    - lamb
    - "1"
  chicken:
    - Chicken
    - hen
    - rooster
    - "2"

keep_classes:
  - cow
  - sheep
  - chicken
```

**Example 3: Performance-Optimized Augmentation**

```yaml
# configs/augment.yaml
max_workers: 8
batch_size: 64
use_optimizations: true
use_cuda: true
memory_pool_size: 100

modes:
  flip_h:
    enabled: true
    prob: 0.5
  # Disable expensive operations for speed
  mosaic:
    enabled: false
  mixup:
    enabled: false
```

### Troubleshooting Configuration Files

**Common Issues**:

1. **YAML Syntax Errors**:

   ```bash
   # Validate YAML syntax
   python -c "import yaml; yaml.safe_load(open('configs/augment.yaml'))"
   ```

2. **Missing Required Keys**:

   - `augment.yaml` must have `modes` section
   - `flag_helper.yaml` must have `aliases` and `keep_classes` sections

3. **Invalid Class Names**:

   - Avoid special characters in class names
   - Use consistent naming conventions

4. **Path Issues**:
   - Use absolute paths or relative paths from dataset root
   - Ensure all referenced directories exist

**Best Practices**:

1. **Backup Original Configs**: Always backup original configuration files before modifying
2. **Test with Small Datasets**: Test configuration changes with small datasets first
3. **Validate Changes**: Use `python main.py help` to verify configuration is loaded correctly
4. **Document Changes**: Keep notes of configuration modifications for reproducibility

---

## Dataset Management

### Dataset Structure

The system expects datasets in YOLO format:

```
dataset/
├── data.yaml          # Dataset configuration
├── images/
│   ├── train/         # Training images
│   ├── val/           # Validation images
│   └── test/          # Test images
└── labels/
    ├── train/         # Training labels (.txt)
    ├── val/           # Validation labels (.txt)
    └── test/          # Test labels (.txt)
```

### Dataset Configuration (`data.yaml`)

```yaml
path: /path/to/dataset
train: images/train
val: images/val
test: images/test

nc: 4 # Number of classes
names: ["cow", "sheep", "chicken", "pig"]
```

### Label Format

YOLO labels are text files with one object per line:

```
class_id center_x center_y width height
```

All coordinates are normalized (0-1).

### Dataset Compilation

The system automatically handles:

- **Class mapping**: Standardizes different naming conventions
- **Duplicate detection**: Removes duplicate images
- **Directory structure**: Maintains proper YOLO format
- **Statistics**: Generates per-class counts and manifests

---

## Training & Evaluation

### Training Process

1. **Dataset Preparation:**

   ```bash
   # Compile multiple datasets
   python main.py compile --compile-inputs ./cow_data ./sheep_data --compile-out ./unified
   ```

2. **Model Training:**

   ```bash
   # Train with compiled dataset
   python main.py train --data ./unified --epochs 100 --size s
   ```

3. **Training Outputs:**
   - Model weights: `./runs/train/train/weights/best.pt`
   - Training logs: `./runs/train/train/`
   - Metrics: `./runs/train/train/results.csv`

### Model Sizes

- **s (small)**: Fastest, lower accuracy
- **m (medium)**: Balanced speed/accuracy
- **x (large)**: Highest accuracy, slower

### Evaluation Metrics

The system provides comprehensive evaluation including:

- **mAP@0.5**: Mean Average Precision at IoU 0.5
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

---

## Data Augmentation

### Augmentation Types

1. **Geometric Transformations:**

   - Horizontal flip
   - Rotation (small angles)
   - Translation
   - Scale/zoom
   - Shear

2. **Color/Visual:**

   - HSV jitter
   - Blur effects
   - Sharpening
   - Grayscale conversion
   - CLAHE enhancement

3. **Advanced Techniques:**
   - Mosaic composition
   - MixUp blending
   - CutOut occlusion

### Configuration

All augmentation settings are controlled via `configs/augment.yaml`:

```yaml
# Enable/disable specific augmentations
modes:
  flip_h:
    enabled: true
    prob: 0.5
  rotate:
    enabled: true
    prob: 0.3
    max_deg: 12
```

### Performance Optimization

- **Parallel processing**: Multi-worker augmentation
- **Memory pooling**: Efficient memory management
- **CUDA acceleration**: GPU-accelerated operations
- **Async I/O**: Non-blocking file operations

---

## Inference & Detection

### Image Inference

```bash
# Run inference on single image
python main.py eval --data ./image.jpg --model ./model/default.pt --conf 0.25 --output ./runs/eval
```

### Video Inference

```bash
# Process video with tracking
python main.py eval --data ./video.mp4 --model ./model/default.pt --fps 15 --tracker bt --conf 0.25 --output ./runs/eval
```

### Batch Processing

```bash
# Process directory of images
python main.py eval --data ./data_folder --model ./model/default.pt --fps 15 --tracker bt --conf 0.25 --output ./runs/eval
```

### Output Formats

- **Images**: Annotated images with bounding boxes and labels saved as `.jpg` files
- **Videos**: Processed videos with object tracking saved as `.mp4` files
- **Results**: Detailed detection results and statistics saved as `.yaml` files

---

## Troubleshooting

### Common Issues

1. **CUDA/GPU Issues:**

   ```bash
   # Check GPU availability
   python -c "import torch; print(torch.cuda.is_available())"

   # Force CPU usage
   python main.py train --data ./dataset --device cpu
   ```

2. **Memory Issues:**

   ```bash
   # Reduce batch size
   python main.py train --data ./dataset --batch 8

   # Use smaller model
   python main.py train --data ./dataset --size s
   ```

3. **Dataset Format Issues:**

   ```bash
   # Validate dataset structure
   python main.py labels --data ./dataset --old-label dummy --new-label dummy
   ```

4. **Dependency Issues:**

   ```bash
   # Reinstall requirements
   pip install -r requirements.txt

   # Check installation
   python main.py help
   ```

### Performance Optimization

1. **GPU Acceleration:**

   - Ensure CUDA is properly installed
   - Use `--device 0` for GPU training
   - Enable CUDA in augmentation config

2. **Memory Management:**

   - Adjust batch size based on available RAM
   - Use memory pooling in augmentation
   - Enable optimizations in config

3. **Parallel Processing:**
   - Set appropriate worker count
   - Use multi-GPU training when available
   - Enable async I/O operations

---

## Examples & Use Cases

### Example 1: Complete Training Pipeline

```bash
# 1. Compile multiple datasets
python main.py compile \
  --compile-inputs ./cow_data ./sheep_data ./chicken_data \
  --compile-out ./livestock_unified \
  --dataset-shuffle

# 2. Augment the compiled dataset
python main.py augment \
  --inputs ./livestock_unified \
  --config ./configs/augment.yaml \
  --workers 4

# 3. Train the model
python main.py train \
  --data ./livestock_unified \
  --epochs 100 \
  --size m \
  --batch 16 \
  --preset balanced

# 4. Run inference on test data
python main.py eval \
  --data ./test_dataset \
  --model ./runs/train/train/weights/best.pt
```

### Example 2: Label Standardization

```bash
# Standardize labels across datasets
python main.py labels --data ./cow_dataset --old-label cattle --new-label cow
python main.py labels --data ./sheep_dataset --old-label ewe --new-label sheep
python main.py labels --data ./chicken_dataset --old-label "0" --new-label chicken

# Compile standardized datasets
python main.py compile \
  --compile-inputs ./cow_dataset ./sheep_dataset ./chicken_dataset \
  --compile-out ./standardized_livestock
```

### Example 3: Production Inference

```bash
# Process farm surveillance video
python main.py eval \
  --data ./farm_surveillance.mp4 \
  --model ./model/default.pt \
  --fps 15 \
  --tracker bt \
  --conf 0.25 \
  --output ./runs/eval

# Batch process images
python main.py eval \
  --data ./farm_images/ \
  --model ./model/default.pt \
  --conf 0.25 \
  --output ./runs/eval

# Process single image with custom confidence
python main.py eval \
  --data ./test_image.jpg \
  --model ./model/default.pt \
  --conf 0.5 \
  --output ./runs/eval
```

### Example 4: Hardware Recommendations

```bash
# Get system recommendations
python main.py recommend

# Output example:
# 💻 HARDWARE CONFIGURATION RECOMMENDATIONS
# ==================================================
#
# 🔍 YOUR CURRENT SYSTEM:
# -------------------------
#    CPU: Intel Core i7-10700K (8 cores, 16 threads)
#    RAM: 32.0 GB
#    GPU: NVIDIA GeForce RTX 3080 (10.0 GB VRAM)
#
# 📊 RECOMMENDED CONFIGURATIONS:
# --------------------------------
#
# 🚀 Quick Preset (Fast Training):
#    Batch Size: 32
#    Model Size: s
#    Workers: 8
#    Estimated Time: 2-3 hours
#
# ⚖️ Balanced Preset (Recommended):
#    Batch Size: 16
#    Model Size: m
#    Workers: 6
#    Estimated Time: 4-6 hours
#
# 💪 Strong Preset (Best Quality):
#    Batch Size: 8
#    Model Size: x
#    Workers: 4
#    Estimated Time: 8-12 hours
```

---

## Additional Resources

### File Structure

```
user_system/
├── main.py                 # Main CLI interface
├── train.py               # Training module
├── inference.py           # Inference module
├── augment.py             # Augmentation module
├── evaluate.py            # Evaluation module
├── configs/               # Configuration files
│   ├── augment.yaml      # Augmentation settings
│   └── flag_helper.yaml  # Class mapping
├── utils/                 # Utility modules
│   ├── arg_helper.py     # Argument processing
│   ├── ui_helper.py      # User interface
│   ├── modify_helper.py  # Label modification
│   └── hardware_info.py  # System information
├── dataset/               # Dataset compilation
├── tests/                 # Test datasets
└── runs/                  # Training outputs
```

### Support

For additional help:

1. Run `python main.py help` for command examples
2. Check configuration files in `configs/`
3. Review test datasets in `tests/`
4. Examine training outputs in `runs/`

---

_This manual covers the complete YOLO Livestock Detection System. For updates and additional features, refer to the project documentation._
