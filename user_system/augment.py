"""
YOLO Dataset Augmentation Module
===============================

This module handles the core augmentation logic for YOLO datasets including:
- Augmentation environment setup coordination
- Image and label augmentation processing
- Augmentation pipeline management
- Individual augmentation transformations (flip, rotate, scale, etc.)
"""

import hashlib
import random
import math
import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
import asyncio
import aiofiles
from queue import Queue
import threading
from dataclasses import dataclass
import warnings
from utils.directory_helper import (
    create_output_directory,
    create_augmented_dataset_structure,
    copy_data_yaml,
    get_base_directory,
    validate_dataset_structure
)




def setup_augmentation_environment(data_paths: List[Path], cfg: Dict[str, Any], base_dir: Path) -> Optional[List[Path]]:
    """
    Set up the complete augmentation environment including output directories and dataset structures.
    
    Args:
        data_paths: List of input dataset paths
        cfg: Configuration dictionary from augment.yaml
        base_dir: Base directory where main.py is located (for output directory creation)
        
    Returns:
        Optional[List[Path]]: List of created augmented dataset paths, or None if setup failed
    """
    print("\n🚀 SETTING UP AUGMENTATION ENVIRONMENT")
    print("=" * 50)
    
    # Determine output directory name
    output_name = cfg.get('output', 'aug_output')
    if not output_name:
        output_name = 'aug_output'
    
    try:
        # Step 1: Create main output directory
        output_dir = create_output_directory(output_name, base_dir)
        
        # Step 2: Create augmented dataset structures
        created_datasets = []
        
        for input_path in data_paths:
            print(f"\n📂 Processing input dataset: {input_path}")
            
            # Create augmented dataset structure
            aug_dataset_path = create_augmented_dataset_structure(input_path, output_dir)
            
            # Copy data.yaml file
            yaml_copied = copy_data_yaml(input_path, aug_dataset_path)
            
            if yaml_copied:
                created_datasets.append(aug_dataset_path)
                print(f"   ✅ Setup complete for: {aug_dataset_path.name}")
            else:
                print(f"   ⚠️  Setup completed with warnings for: {aug_dataset_path.name}")
                created_datasets.append(aug_dataset_path)  # Still add it, just with warning
        
        print(f"\n🎉 Environment setup complete!")
        print(f"   Output directory: {output_dir}")
        print(f"   Created {len(created_datasets)} augmented dataset structures")
        
        return created_datasets
        
    except Exception as e:
        print(f"\n❌ Failed to set up augmentation environment")
        print(f"   Error: {e}")
        return None


# =============================================================================
# PERFORMANCE OPTIMIZATION CLASSES
# =============================================================================

@dataclass
class ImageBuffer:
    """Reusable image buffer for memory pooling."""
    data: np.ndarray
    in_use: bool = False
    max_shape: Tuple[int, int, int] = (0, 0, 0)


class ImageMemoryPool:
    """Memory pool for reusing image buffers to reduce allocation overhead."""
    
    def __init__(self, pool_size: int = 50, max_image_size: Tuple[int, int, int] = (2048, 2048, 3)):
        self.pool_size = pool_size
        self.max_image_size = max_image_size
        self.available_buffers = Queue()
        self.all_buffers = []
        self._initialize_pool()
    
    def _initialize_pool(self):
        """Initialize the memory pool with pre-allocated buffers."""
        for _ in range(self.pool_size):
            buffer_data = np.empty(self.max_image_size, dtype=np.uint8)
            buffer = ImageBuffer(data=buffer_data, max_shape=self.max_image_size)
            self.all_buffers.append(buffer)
            self.available_buffers.put(buffer)
    
    def get_buffer(self, required_shape: Tuple[int, int, int]) -> Optional[np.ndarray]:
        """
        Get a buffer from the pool that can accommodate the required shape.
        
        Args:
            required_shape: (height, width, channels) required
            
        Returns:
            NumPy array buffer or None if no suitable buffer available
        """
        try:
            buffer = self.available_buffers.get_nowait()
            if all(buffer.max_shape[i] >= required_shape[i] for i in range(3)):
                buffer.in_use = True
                # Return a view of the buffer with the required shape
                return buffer.data[:required_shape[0], :required_shape[1], :required_shape[2]]
            else:
                # Buffer too small, put it back and return None
                self.available_buffers.put(buffer)
                return None
        except:
            return None
    
    def return_buffer(self, buffer_view: np.ndarray):
        """
        Return a buffer to the pool.
        
        Args:
            buffer_view: The buffer view to return
        """
        # Find the original buffer that contains this view
        for buffer in self.all_buffers:
            if buffer.in_use and np.shares_memory(buffer.data, buffer_view):
                buffer.in_use = False
                self.available_buffers.put(buffer)
                break


class AsyncImageLoader:
    """Asynchronous image loader with prefetching capabilities."""
    
    def __init__(self, memory_pool: ImageMemoryPool, prefetch_size: int = 10):
        self.memory_pool = memory_pool
        self.prefetch_size = prefetch_size
        self.load_cache = {}
        
    async def load_image_async(self, image_path: Path) -> Optional[np.ndarray]:
        """
        Asynchronously load an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Loaded image as NumPy array or None if failed
        """
        try:
            # Read file asynchronously
            async with aiofiles.open(image_path, 'rb') as f:
                image_bytes = await f.read()
            
            # Decode image (this is still synchronous, but fast)
            image_array = np.frombuffer(image_bytes, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if image is None:
                return None
                
            return image
            
        except Exception as e:
            print(f"⚠️  Async load failed for {image_path}: {e}")
            return None
    
    async def load_batch_async(self, image_paths: List[Path]) -> List[Optional[np.ndarray]]:
        """
        Load multiple images asynchronously.
        
        Args:
            image_paths: List of image paths to load
            
        Returns:
            List of loaded images (None for failed loads)
        """
        tasks = [self.load_image_async(path) for path in image_paths]
        return await asyncio.gather(*tasks, return_exceptions=False)


# =============================================================================
# NUMPY BOUNDING BOX OPERATIONS
# =============================================================================

def convert_boxes_to_numpy(boxes: List[List[float]]) -> np.ndarray:
    """
    Convert list of bounding boxes to NumPy array for faster operations.
    
    Args:
        boxes: List of [class_id, x_center, y_center, width, height]
        
    Returns:
        NumPy array of shape (N, 5) with dtype float32
    """
    if not boxes:
        return np.empty((0, 5), dtype=np.float32)
    return np.array(boxes, dtype=np.float32)


def convert_boxes_to_list(boxes_array: np.ndarray) -> List[List[float]]:
    """
    Convert NumPy array of bounding boxes back to list format.
    
    Args:
        boxes_array: NumPy array of shape (N, 5)
        
    Returns:
        List of [class_id, x_center, y_center, width, height]
    """
    return boxes_array.tolist()


def clip_boxes_numpy(boxes: np.ndarray) -> np.ndarray:
    """
    Clip bounding box coordinates to valid range [0, 1] using NumPy operations.
    
    Args:
        boxes: NumPy array of shape (N, 5) [class_id, x_center, y_center, width, height]
        
    Returns:
        Clipped bounding boxes
    """
    if boxes.size == 0:
        return boxes
    
    # Create a copy to avoid modifying original
    clipped = boxes.copy()
    
    # Clip coordinates (columns 1-4) to [0, 1] range
    clipped[:, 1:] = np.clip(clipped[:, 1:], 0.0, 1.0)
    
    return clipped


def transform_boxes_flip_horizontal_numpy(boxes: np.ndarray) -> np.ndarray:
    """
    Transform bounding boxes for horizontal flip using NumPy operations.
    
    Args:
        boxes: NumPy array of shape (N, 5)
        
    Returns:
        Transformed bounding boxes
    """
    if boxes.size == 0:
        return boxes
    
    transformed = boxes.copy()
    # Flip x_center: new_x = 1.0 - x_center (column 1)
    transformed[:, 1] = 1.0 - transformed[:, 1]
    
    return clip_boxes_numpy(transformed)


def transform_boxes_translate_numpy(boxes: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Transform bounding boxes for translation using NumPy operations.
    
    Args:
        boxes: NumPy array of shape (N, 5)
        dx: Translation in x direction (normalized)
        dy: Translation in y direction (normalized)
        
    Returns:
        Transformed bounding boxes
    """
    if boxes.size == 0:
        return boxes
    
    transformed = boxes.copy()
    # Translate centers (columns 1 and 2)
    transformed[:, 1] += dx  # x_center
    transformed[:, 2] += dy  # y_center
    
    return clip_boxes_numpy(transformed)


def transform_boxes_scale_numpy(boxes: np.ndarray, scale_x: float, scale_y: float, 
                               img_center_x: float = 0.5, img_center_y: float = 0.5) -> np.ndarray:
    """
    Transform bounding boxes for scaling using NumPy operations.
    
    Args:
        boxes: NumPy array of shape (N, 5)
        scale_x: Scale factor in x direction
        scale_y: Scale factor in y direction
        img_center_x: Image center x coordinate (normalized)
        img_center_y: Image center y coordinate (normalized)
        
    Returns:
        Transformed bounding boxes
    """
    if boxes.size == 0:
        return boxes
    
    transformed = boxes.copy()
    
    # Scale positions relative to image center
    transformed[:, 1] = img_center_x + (transformed[:, 1] - img_center_x) / scale_x  # x_center
    transformed[:, 2] = img_center_y + (transformed[:, 2] - img_center_y) / scale_y  # y_center
    
    # Scale dimensions
    transformed[:, 3] /= scale_x  # width
    transformed[:, 4] /= scale_y  # height
    
    return clip_boxes_numpy(transformed)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def generate_file_hash(image_path: Path, label_path: Path) -> str:
    """
    Generate a unique hash for an image-label pair to avoid filename collisions.
    Handles missing label files gracefully.
    
    Args:
        image_path: Path to the image file
        label_path: Path to the label file (may not exist)
        
    Returns:
        str: 8-character hash string
    """
    try:
        # Get image size
        image_size = image_path.stat().st_size
        
        # Get label size (0 if file doesn't exist)
        label_size = label_path.stat().st_size if label_path.exists() else 0
        
        # Combine paths and file sizes for uniqueness
        content = f"{image_path.name}_{label_path.name}_{image_size}_{label_size}"
        hash_object = hashlib.md5(content.encode())
        return hash_object.hexdigest()[:8]
        
    except Exception as e:
        # Fallback: use just the image path for hashing
        print(f"⚠️  Hash generation fallback for {image_path.name}: {e}")
        content = f"{image_path.name}_{image_path.parent.name}"
        hash_object = hashlib.md5(content.encode())
        return hash_object.hexdigest()[:8]


def load_yolo_labels(label_path: Path) -> np.ndarray:
    """
    Load YOLO format labels from a text file as NumPy array.
    Handles empty files and missing files gracefully.
    
    Args:
        label_path: Path to the label file
        
    Returns:
        NumPy array of shape (N, 5) with bounding boxes [class_id, x_center, y_center, width, height]
        Returns empty array (0, 5) for images without labels (which is valid)
    """
    if not label_path.exists():
        # Missing label file - this is valid for background images
        return np.empty((0, 5), dtype=np.float32)
    
    try:
        # Check if file is empty first
        if label_path.stat().st_size == 0:
            # Empty label file - this is valid for background images
            return np.empty((0, 5), dtype=np.float32)
        
        # Use numpy.loadtxt for faster loading
        boxes = np.loadtxt(label_path, dtype=np.float32, ndmin=2)
        
        # Handle the case where loadtxt returns empty array
        if boxes.size == 0:
            return np.empty((0, 5), dtype=np.float32)
        
        # Ensure we have the right shape
        if boxes.ndim == 1:
            # Single row case
            if len(boxes) >= 5:
                return boxes[:5].reshape(1, -1)
            else:
                return np.empty((0, 5), dtype=np.float32)
        elif boxes.ndim == 2:
            # Multiple rows case
            if boxes.shape[1] >= 5:
                return boxes[:, :5]  # Take only first 5 columns
            else:
                return np.empty((0, 5), dtype=np.float32)
        else:
            return np.empty((0, 5), dtype=np.float32)
            
    except Exception as e:
        # Handle malformed files gracefully - still process the image
        if "no data" in str(e).lower() or "empty" in str(e).lower():
            # Empty file - valid case for background images
            return np.empty((0, 5), dtype=np.float32)
        else:
            # Other errors (like column mismatch) - log but continue
            print(f"⚠️  Label file format issue in {label_path}: {e}")
            print(f"   → Treating as background image (no labels)")
            return np.empty((0, 5), dtype=np.float32)


def save_yolo_labels(boxes: np.ndarray, label_path: Path) -> bool:
    """
    Save YOLO format labels to a text file from NumPy array.
    
    Args:
        boxes: NumPy array of shape (N, 5) with bounding boxes [class_id, x_center, y_center, width, height]
        label_path: Path where to save the labels
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        label_path.parent.mkdir(parents=True, exist_ok=True)
        
        if boxes.size == 0:
            # Create empty file
            label_path.touch()
            return True
        
        # Use numpy.savetxt for faster saving
        # Format: class_id as int, coordinates as float with 6 decimal places
        np.savetxt(label_path, boxes, fmt='%d %.6f %.6f %.6f %.6f')
        return True
        
    except Exception as e:
        print(f"❌ Error saving labels to {label_path}: {e}")
        return False


def copy_original_pair(image_path: Path, label_path: Path, output_dir: Path, 
                      split: str, file_hash: str) -> Tuple[Path, Path]:
    """
    Copy original image-label pair to output directory with hash-based naming.
    
    Args:
        image_path: Source image path
        label_path: Source label path
        output_dir: Output dataset directory
        split: Dataset split ('train', 'test', 'valid')
        file_hash: Hash string for the file pair
        
    Returns:
        Tuple of (new_image_path, new_label_path)
    """
    # Create destination paths
    image_ext = image_path.suffix
    new_image_path = output_dir / split / "images" / f"{file_hash}_orig{image_ext}"
    new_label_path = output_dir / split / "labels" / f"{file_hash}_orig.txt"
    
    # Copy files
    try:
        import shutil
        shutil.copy2(image_path, new_image_path)
        shutil.copy2(label_path, new_label_path)
        return new_image_path, new_label_path
    except Exception as e:
        print(f"❌ Error copying original pair: {e}")
        raise


# =============================================================================
# BOUNDING BOX TRANSFORMATION UTILITIES
# =============================================================================

def clip_bbox(bbox: List[float]) -> List[float]:
    """
    Clip bounding box coordinates to valid range [0, 1].
    
    Args:
        bbox: [class_id, x_center, y_center, width, height]
        
    Returns:
        Clipped bounding box
    """
    class_id, x_center, y_center, width, height = bbox
    
    # Clip coordinates to [0, 1] range
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    width = max(0.0, min(1.0, width))
    height = max(0.0, min(1.0, height))
    
    return [class_id, x_center, y_center, width, height]


def transform_bbox_flip_horizontal(bbox: List[float]) -> List[float]:
    """
    Transform bounding box for horizontal flip.
    
    Args:
        bbox: [class_id, x_center, y_center, width, height]
        
    Returns:
        Transformed bounding box
    """
    class_id, x_center, y_center, width, height = bbox
    # Flip x_center: new_x = 1.0 - x_center
    new_x_center = 1.0 - x_center
    return clip_bbox([class_id, new_x_center, y_center, width, height])


def transform_bbox_translate(bbox: List[float], dx: float, dy: float) -> List[float]:
    """
    Transform bounding box for translation.
    
    Args:
        bbox: [class_id, x_center, y_center, width, height]
        dx: Translation in x direction (normalized)
        dy: Translation in y direction (normalized)
        
    Returns:
        Transformed bounding box
    """
    class_id, x_center, y_center, width, height = bbox
    new_x_center = x_center + dx
    new_y_center = y_center + dy
    return clip_bbox([class_id, new_x_center, new_y_center, width, height])


def transform_bbox_scale(bbox: List[float], scale_x: float, scale_y: float, 
                        img_center_x: float = 0.5, img_center_y: float = 0.5) -> List[float]:
    """
    Transform bounding box for scaling around image center.
    
    Args:
        bbox: [class_id, x_center, y_center, width, height]
        scale_x: Scale factor in x direction
        scale_y: Scale factor in y direction
        img_center_x: Image center x coordinate (normalized)
        img_center_y: Image center y coordinate (normalized)
        
    Returns:
        Transformed bounding box
    """
    class_id, x_center, y_center, width, height = bbox
    
    # Scale position relative to image center
    new_x_center = img_center_x + (x_center - img_center_x) / scale_x
    new_y_center = img_center_y + (y_center - img_center_y) / scale_y
    
    # Scale dimensions
    new_width = width / scale_x
    new_height = height / scale_y
    
    return clip_bbox([class_id, new_x_center, new_y_center, new_width, new_height])


# =============================================================================
# INDIVIDUAL AUGMENTATION FUNCTIONS
# =============================================================================

def augment_flip_horizontal(image: np.ndarray, boxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply horizontal flip augmentation with NumPy optimization.
    
    Args:
        image: Input image as numpy array
        boxes: NumPy array of bounding boxes (N, 5)
        
    Returns:
        Tuple of (augmented_image, transformed_boxes)
    """
    # Flip image horizontally
    flipped_image = cv2.flip(image, 1)
    
    # Transform bounding boxes using vectorized NumPy operations
    transformed_boxes = transform_boxes_flip_horizontal_numpy(boxes)
    
    return flipped_image, transformed_boxes


def augment_rotate(image: np.ndarray, boxes: List[List[float]], max_angle: float) -> Tuple[np.ndarray, List[List[float]]]:
    """
    Apply rotation augmentation.
    
    Args:
        image: Input image as numpy array
        boxes: List of bounding boxes
        max_angle: Maximum rotation angle in degrees
        
    Returns:
        Tuple of (augmented_image, transformed_boxes)
    """
    height, width = image.shape[:2]
    angle = random.uniform(-max_angle, max_angle)
    
    # Get rotation matrix
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Apply rotation
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    
    # For simplicity, keep original boxes (rotation transformation is complex)
    # In a production system, you'd want proper bbox rotation
    transformed_boxes = [clip_bbox(box) for box in boxes]
    
    return rotated_image, transformed_boxes


def augment_translate(image: np.ndarray, boxes: np.ndarray, max_fraction: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply translation augmentation with NumPy optimization.
    
    Args:
        image: Input image as numpy array
        boxes: NumPy array of bounding boxes (N, 5)
        max_fraction: Maximum translation as fraction of image size
        
    Returns:
        Tuple of (augmented_image, transformed_boxes)
    """
    height, width = image.shape[:2]
    
    # Random translation
    dx_pixels = random.uniform(-max_fraction * width, max_fraction * width)
    dy_pixels = random.uniform(-max_fraction * height, max_fraction * height)
    
    # Create translation matrix
    translation_matrix = np.float32([[1, 0, dx_pixels], [0, 1, dy_pixels]])
    
    # Apply translation
    translated_image = cv2.warpAffine(image, translation_matrix, (width, height))
    
    # Transform bounding boxes using vectorized NumPy operations
    dx_norm = dx_pixels / width
    dy_norm = dy_pixels / height
    transformed_boxes = transform_boxes_translate_numpy(boxes, dx_norm, dy_norm)
    
    return translated_image, transformed_boxes


def augment_scale(image: np.ndarray, boxes: List[List[float]], scale_range: List[float]) -> Tuple[np.ndarray, List[List[float]]]:
    """
    Apply scaling augmentation.
    
    Args:
        image: Input image as numpy array
        boxes: List of bounding boxes
        scale_range: [min_scale, max_scale] range
        
    Returns:
        Tuple of (augmented_image, transformed_boxes)
    """
    height, width = image.shape[:2]
    scale = random.uniform(scale_range[0], scale_range[1])
    
    # Calculate new dimensions
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Resize image
    scaled_image = cv2.resize(image, (new_width, new_height))
    
    # Crop or pad to original size
    if scale > 1.0:
        # Crop from center
        start_x = (new_width - width) // 2
        start_y = (new_height - height) // 2
        final_image = scaled_image[start_y:start_y + height, start_x:start_x + width]
    else:
        # Pad to original size
        final_image = np.zeros((height, width, image.shape[2]), dtype=image.dtype)
        start_x = (width - new_width) // 2
        start_y = (height - new_height) // 2
        final_image[start_y:start_y + new_height, start_x:start_x + new_width] = scaled_image
    
    # Transform bounding boxes
    transformed_boxes = [transform_bbox_scale(box, scale, scale) for box in boxes]
    
    return final_image, transformed_boxes


def augment_hsv(image: np.ndarray, boxes: List[List[float]], h_gain: float, s_gain: float, v_gain: float) -> Tuple[np.ndarray, List[List[float]]]:
    """
    Apply HSV color augmentation.
    
    Args:
        image: Input image as numpy array
        boxes: List of bounding boxes
        h_gain: Hue gain factor
        s_gain: Saturation gain factor
        v_gain: Value gain factor
        
    Returns:
        Tuple of (augmented_image, unchanged_boxes)
    """
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    
    # Apply random gains
    h_random = random.uniform(-h_gain, h_gain)
    s_random = random.uniform(1 - s_gain, 1 + s_gain)
    v_random = random.uniform(1 - v_gain, 1 + v_gain)
    
    hsv[:, :, 0] = (hsv[:, :, 0] + h_random * 180) % 180  # Hue
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s_random, 0, 255)  # Saturation
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * v_random, 0, 255)  # Value
    
    # Convert back to BGR
    augmented_image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    # Boxes remain unchanged for color augmentation
    return augmented_image, boxes


def augment_blur(image: np.ndarray, boxes: List[List[float]], kernel_sizes: List[int]) -> Tuple[np.ndarray, List[List[float]]]:
    """
    Apply blur augmentation.
    
    Args:
        image: Input image as numpy array
        boxes: List of bounding boxes
        kernel_sizes: List of possible kernel sizes
        
    Returns:
        Tuple of (augmented_image, unchanged_boxes)
    """
    kernel_size = random.choice(kernel_sizes)
    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    # Boxes remain unchanged for blur augmentation
    return blurred_image, boxes


# =============================================================================
# OPTIMIZATION UTILITIES
# =============================================================================

def optimize_image_loading(image_path: Path) -> Optional[np.ndarray]:
    """
    Optimized image loading with error handling.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Loaded image or None if failed
    """
    try:
        # Use cv2.IMREAD_COLOR for faster loading
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        print(f"⚠️  Failed to load {image_path}: {e}")
        return None


def batch_save_images(image_data_list: List[Tuple[np.ndarray, Path]]) -> int:
    """
    Save multiple images in batch for better I/O performance.
    
    Args:
        image_data_list: List of (image, path) tuples
        
    Returns:
        Number of successfully saved images
    """
    saved_count = 0
    for image, path in image_data_list:
        try:
            # Ensure directory exists
            path.parent.mkdir(parents=True, exist_ok=True)
            # Save image
            if cv2.imwrite(str(path), image):
                saved_count += 1
        except Exception as e:
            print(f"⚠️  Failed to save {path}: {e}")
    
    return saved_count


def augment_sharpen(image: np.ndarray, boxes: List[List[float]]) -> Tuple[np.ndarray, List[List[float]]]:
    """
    Apply sharpening augmentation.
    
    Args:
        image: Input image as numpy array
        boxes: List of bounding boxes
        
    Returns:
        Tuple of (augmented_image, unchanged_boxes)
    """
    # Sharpening kernel
    kernel = np.array([[-1, -1, -1],
                      [-1,  9, -1],
                      [-1, -1, -1]])
    
    # Apply sharpening
    sharpened_image = cv2.filter2D(image, -1, kernel)
    
    # Boxes remain unchanged for sharpening augmentation
    return sharpened_image, boxes


def augment_grayscale(image: np.ndarray, boxes: List[List[float]]) -> Tuple[np.ndarray, List[List[float]]]:
    """
    Apply grayscale augmentation.
    
    Args:
        image: Input image as numpy array
        boxes: List of bounding boxes
        
    Returns:
        Tuple of (augmented_image, unchanged_boxes)
    """
    # Convert to grayscale and back to 3-channel
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayscale_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    # Boxes remain unchanged for grayscale augmentation
    return grayscale_image, boxes


# =============================================================================
# MAIN AUGMENTATION ORCHESTRATOR
# =============================================================================

def process_image_pair_worker(args_tuple) -> int:
    """
    Worker function for parallel processing of image pairs.
    
    Args:
        args_tuple: Tuple containing (image_path, label_path, output_dir, split, cfg, file_hash)
        
    Returns:
        int: Number of augmented pairs created
    """
    image_path, label_path, output_dir, split, cfg, file_hash = args_tuple
    return process_image_pair(image_path, label_path, output_dir, split, cfg, file_hash)


def process_image_pair_optimized(image_path: Path, label_path: Path, output_dir: Path, 
                                split: str, cfg: Dict[str, Any], file_hash: str, 
                                memory_pool: Optional[ImageMemoryPool] = None,
                                async_loader: Optional[AsyncImageLoader] = None) -> int:
    """
    Process a single image-label pair with optimized performance.
    
    Args:
        image_path: Path to source image
        label_path: Path to source label file
        output_dir: Output dataset directory
        split: Dataset split ('train', 'test', 'valid')
        cfg: Configuration dictionary
        file_hash: Hash for the file pair
        memory_pool: Optional memory pool for buffer reuse
        async_loader: Optional async loader for I/O optimization
        
    Returns:
        int: Number of augmented pairs created
    """
    try:
        # Load image with optimization
        if async_loader:
            # For async loading, we'd need to restructure the calling code
            # For now, fall back to optimized sync loading
            image = optimize_image_loading(image_path)
        else:
            image = optimize_image_loading(image_path)
            
        if image is None:
            print(f"⚠️  Could not load image: {image_path}")
            return 0
        
        # Load labels as NumPy array
        boxes = load_yolo_labels(label_path)
        
        # Copy original if keep_original is True
        created_count = 0
        if cfg.get('keep_original', True):
            copy_original_pair(image_path, label_path, output_dir, split, file_hash)
            created_count += 1
        
        # Get augmentation settings
        modes = cfg.get('modes', {})
        
        # Handle counts vs multiplier (mutually exclusive)
        counts = cfg.get('counts')
        multiplier = cfg.get('multiplier')
        
        if counts is None and multiplier is not None:
            counts = max(1, int(multiplier))
        elif counts is None:
            counts = 1
        else:
            counts = max(1, int(counts))
            
        prob_default = cfg.get('prob_default', 0.3)
        deterministic = cfg.get('deterministic', False)
        
        # Get list of enabled augmentations
        enabled_augmentations = []
        
        if modes.get('flip_h', {}).get('enabled', False):
            enabled_augmentations.append(('flip_h', modes['flip_h']))
        if modes.get('translate', {}).get('enabled', False):
            enabled_augmentations.append(('translate', modes['translate']))
        # Add other optimized augmentations as needed
        
        # Apply each augmentation individually for 'counts' times
        for count_idx in range(counts):
            for aug_idx, (aug_name, aug_config) in enumerate(enabled_augmentations):
                # Check probability for this augmentation (unless deterministic mode)
                if not deterministic:
                    prob = aug_config.get('prob', prob_default)
                    if random.random() >= prob:
                        continue  # Skip this augmentation based on probability
                
                # Get buffer from memory pool if available
                if memory_pool:
                    buffer = memory_pool.get_buffer(image.shape)
                    if buffer is not None:
                        # Copy image to buffer
                        np.copyto(buffer, image)
                        current_image = buffer
                    else:
                        current_image = image.copy()
                else:
                    current_image = image.copy()
                
                # Work with NumPy array for boxes
                current_boxes = boxes.copy()
                
                # Apply the specific augmentation with NumPy optimization
                if aug_name == 'flip_h':
                    current_image, current_boxes = augment_flip_horizontal(current_image, current_boxes)
                elif aug_name == 'translate':
                    max_frac = aug_config.get('max_frac', 0.1)
                    current_image, current_boxes = augment_translate(current_image, current_boxes, max_frac)
                # Add other optimized augmentations here
                
                # Create unique filename for this specific augmentation
                image_ext = image_path.suffix
                if counts > 1:
                    aug_suffix = f"{aug_name}_c{count_idx + 1}"
                else:
                    aug_suffix = aug_name
                
                aug_image_path = output_dir / split / "images" / f"{file_hash}_{aug_suffix}{image_ext}"
                aug_label_path = output_dir / split / "labels" / f"{file_hash}_{aug_suffix}.txt"
                
                # Save the augmented image and labels
                cv2.imwrite(str(aug_image_path), current_image)
                save_yolo_labels(current_boxes, aug_label_path)
                
                # Return buffer to pool if used
                if memory_pool and buffer is not None:
                    memory_pool.return_buffer(current_image)
                
                created_count += 1
        
        return created_count
        
    except Exception as e:
        print(f"❌ Error processing {image_path}: {e}")
        return 0


def process_image_pair(image_path: Path, label_path: Path, output_dir: Path, 
                      split: str, cfg: Dict[str, Any], file_hash: str) -> int:
    """
    Process a single image-label pair with augmentations.
    
    Args:
        image_path: Path to source image
        label_path: Path to source label file
        output_dir: Output dataset directory
        split: Dataset split ('train', 'test', 'valid')
        cfg: Configuration dictionary
        file_hash: Hash for the file pair
        
    Returns:
        int: Number of augmented pairs created
    """
    try:
        # Load image with optimization
        image = optimize_image_loading(image_path)
        if image is None:
            print(f"⚠️  Could not load image: {image_path}")
            return 0
        
        # Load labels as NumPy array
        boxes = load_yolo_labels(label_path)
        
        # Copy original if keep_original is True
        created_count = 0
        if cfg.get('keep_original', True):
            copy_original_pair(image_path, label_path, output_dir, split, file_hash)
            created_count += 1
        
        # Get augmentation settings
        modes = cfg.get('modes', {})
        
        # Handle counts vs multiplier (mutually exclusive)
        counts = cfg.get('counts')
        multiplier = cfg.get('multiplier')
        
        if counts is None and multiplier is not None:
            counts = max(1, int(multiplier))
        elif counts is None:
            counts = 1
        else:
            counts = max(1, int(counts))
            
        prob_default = cfg.get('prob_default', 0.3)
        deterministic = cfg.get('deterministic', False)
        
        # Get list of enabled augmentations with their configurations
        enabled_augmentations = []
        
        if modes.get('flip_h', {}).get('enabled', False):
            enabled_augmentations.append(('flip_h', modes['flip_h']))
        if modes.get('rotate', {}).get('enabled', False):
            enabled_augmentations.append(('rotate', modes['rotate']))
        if modes.get('translate', {}).get('enabled', False):
            enabled_augmentations.append(('translate', modes['translate']))
        if modes.get('scale', {}).get('enabled', False):
            enabled_augmentations.append(('scale', modes['scale']))
        if modes.get('hsv', {}).get('enabled', False):
            enabled_augmentations.append(('hsv', modes['hsv']))
        if modes.get('blur', {}).get('enabled', False):
            enabled_augmentations.append(('blur', modes['blur']))
        if modes.get('sharpen', {}).get('enabled', False):
            enabled_augmentations.append(('sharpen', modes['sharpen']))
        if modes.get('grayscale', {}).get('enabled', False):
            enabled_augmentations.append(('grayscale', modes['grayscale']))
        
        # Generate 'counts' number of combined augmented images
        for count_idx in range(counts):
            # Start with a fresh copy of the original image and boxes
            current_image = image.copy()
            current_boxes = boxes.copy()
            applied_augs = []
            
            # Apply augmentations based on mode (deterministic vs probabilistic)
            for aug_name, aug_config in enabled_augmentations:
                should_apply = False
                
                if deterministic:
                    # Deterministic mode: apply ALL enabled augmentations
                    should_apply = True
                else:
                    # Probabilistic mode: apply based on probability
                    prob = aug_config.get('prob', prob_default)
                    should_apply = random.random() < prob
                
                if should_apply:
                    # Apply the augmentation to the current image (cumulative effect)
                    if aug_name == 'flip_h':
                        current_image, current_boxes = augment_flip_horizontal(current_image, current_boxes)
                        applied_augs.append('flip_h')
                    elif aug_name == 'translate':
                        max_frac = aug_config.get('max_frac', 0.1)
                        current_image, current_boxes = augment_translate(current_image, current_boxes, max_frac)
                        applied_augs.append('translate')
                    elif aug_name == 'rotate':
                        max_deg = aug_config.get('max_deg', 15)
                        boxes_list = convert_boxes_to_list(current_boxes)
                        current_image, boxes_result = augment_rotate(current_image, boxes_list, max_deg)
                        current_boxes = convert_boxes_to_numpy(boxes_result)
                        applied_augs.append('rotate')
                    elif aug_name == 'scale':
                        scale_range = aug_config.get('range', [0.8, 1.2])
                        boxes_list = convert_boxes_to_list(current_boxes)
                        current_image, boxes_result = augment_scale(current_image, boxes_list, scale_range)
                        current_boxes = convert_boxes_to_numpy(boxes_result)
                        applied_augs.append('scale')
                    elif aug_name == 'hsv':
                        h_gain = aug_config.get('h', 0.015)
                        s_gain = aug_config.get('s', 0.7)
                        v_gain = aug_config.get('v', 0.4)
                        boxes_list = convert_boxes_to_list(current_boxes)
                        current_image, boxes_result = augment_hsv(current_image, boxes_list, h_gain, s_gain, v_gain)
                        current_boxes = convert_boxes_to_numpy(boxes_result)
                        applied_augs.append('hsv')
                    elif aug_name == 'blur':
                        # Use CUDA-accelerated blur if available
                        kernel_sizes = aug_config.get('ksize', [3, 5])
                        current_image, current_boxes = augment_blur_cuda(current_image, current_boxes, kernel_sizes)
                        applied_augs.append('blur')
                    elif aug_name == 'sharpen':
                        boxes_list = convert_boxes_to_list(current_boxes)
                        current_image, boxes_result = augment_sharpen(current_image, boxes_list)
                        current_boxes = convert_boxes_to_numpy(boxes_result)
                        applied_augs.append('sharpen')
                    elif aug_name == 'grayscale':
                        boxes_list = convert_boxes_to_list(current_boxes)
                        current_image, boxes_result = augment_grayscale(current_image, boxes_list)
                        current_boxes = convert_boxes_to_numpy(boxes_result)
                        applied_augs.append('grayscale')
            
            # Save the combined augmented image (only if augmentations were applied or counts=1)
            if applied_augs or counts == 1:
                image_ext = image_path.suffix
                
                # Create descriptive filename
                if applied_augs:
                    aug_suffix = f"aug{count_idx + 1}_{'_'.join(applied_augs[:3])}"  # Limit to 3 for filename length
                    if len(applied_augs) > 3:
                        aug_suffix += f"_plus{len(applied_augs) - 3}"
                else:
                    aug_suffix = f"aug{count_idx + 1}_none"
                
                aug_image_path = output_dir / split / "images" / f"{file_hash}_{aug_suffix}{image_ext}"
                aug_label_path = output_dir / split / "labels" / f"{file_hash}_{aug_suffix}.txt"
                
                # Save the combined augmented image and labels
                cv2.imwrite(str(aug_image_path), current_image)
                save_yolo_labels(current_boxes, aug_label_path)
                
                created_count += 1
        
        return created_count
        
    except Exception as e:
        print(f"❌ Error processing {image_path}: {e}")
        return 0


def find_image_label_pairs(dataset_path: Path, split: str) -> List[Tuple[Path, Path]]:
    """
    Find image-label pairs in a dataset split.
    Includes images without labels (background images).
    
    Args:
        dataset_path: Path to the dataset directory
        split: Dataset split ('train', 'test', 'valid')
        
    Returns:
        List of (image_path, label_path) tuples
        Note: label_path might not exist for background images (will be handled gracefully)
    """
    images_dir = dataset_path / split / "images"
    labels_dir = dataset_path / split / "labels"
    
    if not images_dir.exists():
        return []
    
    # Create labels directory if it doesn't exist
    if not labels_dir.exists():
        labels_dir.mkdir(parents=True, exist_ok=True)
    
    pairs = []
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    for image_path in images_dir.iterdir():
        if image_path.suffix.lower() in image_extensions:
            # Find corresponding label file (or create path for missing one)
            label_name = image_path.stem + '.txt'
            label_path = labels_dir / label_name
            
            # Always add the pair - load_yolo_labels will handle missing/empty files
            pairs.append((image_path, label_path))
    
    return pairs


def run_augmentation_pipeline(data_paths: List[Path], created_datasets: List[Path], 
                             cfg: Dict[str, Any], max_workers: Optional[int] = None) -> Dict[str, Any]:
    """
    Main augmentation pipeline that processes all datasets with parallel processing.
    
    Args:
        data_paths: List of input dataset paths
        created_datasets: List of created augmented dataset paths
        cfg: Configuration dictionary
        max_workers: Maximum number of parallel workers (None = auto-detect)
        
    Returns:
        Dictionary with processing statistics
    """
    print("\n🎨 STARTING AUGMENTATION PIPELINE")
    print("=" * 50)
    
    # Set random seed for reproducibility
    seed = cfg.get('seed')
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        print(f"🎲 Random seed set to: {seed}")
    
    # Determine number of workers
    if max_workers is None:
        max_workers = min(mp.cpu_count(), 8)  # Cap at 8 to avoid memory issues
    print(f"🔧 Using {max_workers} parallel workers")
    
    # Initialize performance optimizations
    use_optimizations = cfg.get('use_optimizations', True)
    memory_pool = None
    async_loader = None
    
    if use_optimizations:
        print(f"⚡ Initializing performance optimizations...")
        # Create memory pool
        pool_size = min(max_workers * 10, 100)  # Scale with workers
        memory_pool = ImageMemoryPool(pool_size=pool_size)
        
        # Create async loader
        async_loader = AsyncImageLoader(memory_pool)
        print(f"   📦 Memory pool: {pool_size} buffers")
        print(f"   🔄 Async I/O: enabled")
    
    total_processed = 0
    total_created = 0
    stats = {}
    
    for input_path, output_path in zip(data_paths, created_datasets):
        print(f"\n📂 Processing dataset: {input_path.name}")
        dataset_stats = {'processed': 0, 'created': 0, 'splits': {}}
        
        # Process each split
        for split in ['train', 'test', 'valid']:
            pairs = find_image_label_pairs(input_path, split)
            if not pairs:
                continue
            
            print(f"   📁 {split}: {len(pairs)} image-label pairs")
            
            # Prepare arguments for parallel processing
            worker_args = []
            for image_path, label_path in pairs:
                file_hash = generate_file_hash(image_path, label_path)
                worker_args.append((image_path, label_path, output_path, split, cfg, file_hash))
            
            # Process pairs in parallel
            split_created = 0
            if len(worker_args) > 1 and max_workers > 1:
                # Use parallel processing for multiple pairs
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    results = list(executor.map(process_image_pair_worker, worker_args))
                    split_created = sum(results)
            else:
                # Sequential processing for single pair or single worker
                for args in worker_args:
                    split_created += process_image_pair_worker(args)
            
            dataset_stats['splits'][split] = {
                'processed': len(pairs),
                'created': split_created
            }
            dataset_stats['processed'] += len(pairs)
            dataset_stats['created'] += split_created
            
            print(f"      ✅ Created {split_created} augmented pairs")
        
        stats[input_path.name] = dataset_stats
        total_processed += dataset_stats['processed']
        total_created += dataset_stats['created']
        
        print(f"   📊 Dataset total: {dataset_stats['processed']} → {dataset_stats['created']} pairs")
    
    print(f"\n🎉 AUGMENTATION PIPELINE COMPLETE!")
    print(f"   📊 Total processed: {total_processed} pairs")
    print(f"   📊 Total created: {total_created} pairs")
    print(f"   📊 Expansion ratio: {total_created / max(total_processed, 1):.2f}x")
    print("=" * 50)
    
    return {
        'total_processed': total_processed,
        'total_created': total_created,
        'expansion_ratio': total_created / max(total_processed, 1),
        'datasets': stats
    }


# =============================================================================
# CUDA SUPPORT DETECTION AND INITIALIZATION
# =============================================================================

class CUDAManager:
    """Manages CUDA availability and GPU operations."""
    
    def __init__(self, silent=False):
        self.cuda_available = False
        self.device_count = 0
        self._silent = silent
        self._initialize_cuda()
    
    def _initialize_cuda(self):
        """Initialize CUDA support detection."""
        try:
            # Check if CUDA is available in OpenCV
            self.device_count = cv2.cuda.getCudaEnabledDeviceCount()
            self.cuda_available = self.device_count > 0
            
            if not self._silent:
                if self.cuda_available:
                    print(f"🎮 CUDA Support: ✅ Enabled ({self.device_count} device(s) detected)")
                    # Test basic CUDA operation
                    try:
                        test_mat = cv2.cuda_GpuMat()
                        print(f"   🔧 CUDA operations: ✅ Functional")
                    except Exception as e:
                        print(f"   ⚠️  CUDA operations: Limited functionality ({e})")
                else:
                    print(f"🎮 CUDA Support: ❌ Not available (using CPU fallback)")
                
        except Exception as e:
            self.cuda_available = False
            self.device_count = 0
            if not self._silent:
                print(f"🎮 CUDA Support: ❌ Detection failed - {e}")
                print(f"   💡 Install opencv-contrib-python-headless for CUDA support")
    
    def is_available(self) -> bool:
        """Check if CUDA is available."""
        return self.cuda_available
    
    def get_device_count(self) -> int:
        """Get number of CUDA devices."""
        return self.device_count


# Global CUDA manager instance (silent during import)
cuda_manager = CUDAManager(silent=True)

def get_cuda_status():
    """Get CUDA status as a formatted message."""
    if cuda_manager.is_available():
        count = cuda_manager.get_device_count()
        return f"🎮 CUDA Support: ✅ Enabled ({count} device(s) detected)"
    return "🎮 CUDA Support: ❌ Not available (using CPU fallback)"


# =============================================================================
# GPU-ACCELERATED AUGMENTATION FUNCTIONS
# =============================================================================

def augment_blur_cuda(image: np.ndarray, boxes: np.ndarray, kernel_sizes: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply blur augmentation using CUDA acceleration if available.
    
    Args:
        image: Input image as numpy array
        boxes: NumPy array of bounding boxes (N, 5)
        kernel_sizes: List of possible kernel sizes
        
    Returns:
        Tuple of (augmented_image, unchanged_boxes)
    """
    if cuda_manager.is_available():
        try:
            # Upload image to GPU
            gpu_image = cv2.cuda_GpuMat()
            gpu_image.upload(image)
            
            # Apply Gaussian blur on GPU
            kernel_size = random.choice(kernel_sizes)
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            gpu_result = cv2.cuda.GaussianBlur(gpu_image, (kernel_size, kernel_size), 0)
            
            # Download result from GPU
            result_image = gpu_result.download()
            
            return result_image, boxes
            
        except Exception as e:
            print(f"⚠️  CUDA blur failed, falling back to CPU: {e}")
            # Fall back to CPU implementation
            boxes_list = convert_boxes_to_list(boxes)
            result_image, result_boxes = augment_blur(image, boxes_list, kernel_sizes)
            return result_image, convert_boxes_to_numpy(result_boxes)
    else:
        # Use CPU implementation
        boxes_list = convert_boxes_to_list(boxes)
        result_image, result_boxes = augment_blur(image, boxes_list, kernel_sizes)
        return result_image, convert_boxes_to_numpy(result_boxes)


def augment_resize_cuda(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Apply resize operation using CUDA acceleration if available.
    
    Args:
        image: Input image as numpy array
        target_size: (width, height) target size
        
    Returns:
        Resized image
    """
    if cuda_manager.is_available():
        try:
            # Upload image to GPU
            gpu_image = cv2.cuda_GpuMat()
            gpu_image.upload(image)
            
            # Resize on GPU
            gpu_result = cv2.cuda.resize(gpu_image, target_size)
            
            # Download result from GPU
            result_image = gpu_result.download()
            
            return result_image
            
        except Exception as e:
            print(f"⚠️  CUDA resize failed, falling back to CPU: {e}")
            return cv2.resize(image, target_size)
    else:
        # Use CPU implementation
        return cv2.resize(image, target_size)


