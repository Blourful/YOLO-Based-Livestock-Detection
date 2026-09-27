"""
YOLO Object Detection and Tracking Module
=========================================

This module provides functionality for detecting and tracking objects in images and videos.

Key Features:
- Single image inference with object classification and counting
- Video inference with configurable frame rate processing (minimum 15fps)
- Batch evaluation on image directories with combined object statistics
- Detailed object type classification for all YOLO model classes
- Object count summaries and tracking
- Peak object detection across video timeline
- Automatic YOLO model loading

Video Processing Features:
- Supports multiple video formats (.mp4, .avi, .mov, .mkv, .flv, .wmv, .webm)
- Configurable target FPS for processing (default: 15fps)
- Intelligent frame skipping to achieve target processing rate
- Comprehensive statistics including total counts and peak simultaneous counts
- Frame-by-frame detailed results and summary reports
- Labeled video output with detection boxes and tracking information
Purpose: General object detection and tracking system for images and videos
"""

import os
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from tqdm import tqdm

MODULE_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = MODULE_ROOT / "runs" / "eval"
DEFAULT_MODEL_PATH = MODULE_ROOT / "model" / "default.pt"

try:
    from ultralytics import YOLO
    import cv2
    import numpy as np
    YOLO_AVAILABLE = True
    CV2_AVAILABLE = True
except ImportError as e:
    YOLO_AVAILABLE = False
    CV2_AVAILABLE = False
    if "ultralytics" in str(e):
        print("⚠️  Ultralytics YOLO not installed. Please install with: pip install ultralytics")
    if "cv2" in str(e):
        print("⚠️  OpenCV not installed. Please install with: pip install opencv-python")


def print_evaluation_results(results):
    """
    Print a unified summary of evaluation results for all processing types.
    
    Args:
        results: Dictionary containing evaluation results in unified format
    """
    if not isinstance(results, dict):
        return
    
    # Print total animals detected
    total = results.get("total_animals_detected", 0)
    print(f"✅ Total Animals Detected: {total}")
    
    # Print animal breakdown
    animal_counts = results.get("animal_counts", {})
    if animal_counts:
        print("🐾 Animal Classification:")
        for animal_type, count in animal_counts.items():
            print(f"   {animal_type}: {count}")
    else:
        print("🔍 No animals detected")
        
    visualization_path = results.get("visualization_path")
    if visualization_path:
        print(f"🎨 Visualization saved to: {visualization_path}")
    
    output_path = results.get("output_path")
    if output_path:
        print(f"📄 Detailed results saved to: {output_path}")
        
    

    
    # Print output location
    # output_path = results.get("output_path")
    # if output_path:
    #     print(f"📁 Detailed results saved to: {output_path}")


def draw_detection_labels(frame, detections, show_track_id=True):
    """
    Draw detection boxes and labels on a frame.
    
    Args:
        frame: OpenCV frame (numpy array)
        detections: List of detection dictionaries
        show_track_id: Whether to show tracking ID in labels
    
    Returns:
        Frame with drawn annotations
    """
    if not CV2_AVAILABLE:
        return frame
    
    annotated_frame = frame.copy()
    
    # Define colors for different animal types
    colors = {
        'cow': (255, 0, 0),      # Blue
        'sheep': (0, 0, 255),    # Red
        'chicken': (0, 255, 255),    # Yellow
        'pig': (255, 255, 0),    # Cyan
        'default': (128, 128, 128) # Gray
    }
    
    for detection in detections:
        bbox = detection['bbox']  # [x1, y1, x2, y2]
        animal_type = detection['animal_type']  # Changed back to 'animal_type'
        confidence = detection['confidence']
        track_id = detection.get('track_id', -1)
        
        # Get color for this animal type (use default for unknown types)
        color = colors.get(animal_type, colors['default'])
        
        # Draw bounding box
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        
        # Prepare label text
        if show_track_id and track_id != -1:
            label = f"{animal_type} ID:{track_id} {confidence:.2f}"
        else:
            label = f"{animal_type} {confidence:.2f}"
        
        # Get text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        # Draw background rectangle for text
        cv2.rectangle(annotated_frame, (x1, y1 - text_height - baseline - 5), 
                     (x1 + text_width, y1), color, -1)
        
        # Draw text
        cv2.putText(annotated_frame, label, (x1, y1 - baseline - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return annotated_frame

def evaluate_model(
    data_path: str,
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
    model_name: str = str(DEFAULT_MODEL_PATH),
    target_fps: int = 15,
    tracker: str = "bt",
    confidence: Optional[float] = None
) -> Dict[str, Any]:
    """
    Main evaluation function that handles single images, directories, and videos.
    
    Args:
        data_path: Path to image file, video file, or directory
        weights: Path to model weights (temporarily ignored, uses YOLOv8n)
        output_dir: Output directory for results
        metrics: List of metrics to compute (temporarily ignored)
    target_fps: Target FPS for video processing (default: 15)
    tracker: Tracker type for video inference
    confidence: Minimum confidence threshold (uses YOLO default when None)
        
    Returns:
        Evaluation results dictionary
    """
    if not YOLO_AVAILABLE:
        raise ImportError("Ultralytics YOLO is required. Install with: pip install ultralytics")

    # Determine evaluation mode first
    mode_dict = {0: "Unknown", 1: "Single Image", 2: "Batch", 3: "Video"}
    evaluation_mode = 0
    data_path = Path(data_path)
    
    if data_path.is_file():
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
        if data_path.suffix.lower() in video_extensions:
            evaluation_mode = 3
        else:
            evaluation_mode = 1
            
    elif data_path.is_dir():
        evaluation_mode = 2
        
    else:
        raise ValueError(f"Invalid data path: {data_path}")
    
    print(f"Inference Mode: {mode_dict[evaluation_mode]} ({data_path})")
    print("✅ Pass")
    print()
    
    print(f"Loading Model: {model_name}")
    try:
        model = YOLO(model_name)
        print("✅ Pass")
    except Exception as e:
        raise e
    
    print()
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if evaluation_mode == 1:
        results = _run_single_image(model, data_path, output_dir, confidence)
    elif evaluation_mode == 2:
        results = _run_batch_evaluation(model, data_path, output_dir, target_fps, tracker, confidence)
    elif evaluation_mode == 3:
        results = _run_video_inference(model, data_path, output_dir, target_fps, tracker, confidence)
    else:
        raise ValueError("Could not determine evaluation mode")
    
    # Print success message and results summary
    print(f"\n🎉 EVALUATION COMPLETED SUCCESSFULLY!")
    print_evaluation_results(results)
    
    return results

def _run_single_image(model, image_path: Path, output_dir: Path, confidence: Optional[float]) -> Dict[str, Any]:
    """Run inference on a single image."""
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Determine the effective confidence threshold used for this inference
    model_default_conf = getattr(model, "overrides", {}).get("conf")
    if model_default_conf is None:
        model_default_conf = 0.25
    else:
        model_default_conf = float(model_default_conf)

    if confidence is not None:
        applied_confidence = float(confidence)
        inference_kwargs = {"conf": applied_confidence}
    else:
        applied_confidence = model_default_conf
        inference_kwargs = {}

    # Run inference using provided confidence threshold when supplied
    results = model(str(image_path), verbose=False, **inference_kwargs)
    result = results[0]
 
    # Count detections and classify animals
    num_detections = len(result.boxes) if result.boxes is not None else 0
    animal_counts = {}
    animal_details = []
    
    if result.boxes is not None and len(result.boxes) > 0:
        # Get class names and counts
        class_ids = result.boxes.cls.cpu().numpy() if result.boxes.cls is not None else []
        confidences = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else []
        
        for i, (class_id, confidence) in enumerate(zip(class_ids, confidences)):
            class_name = model.names[int(class_id)]
            
            # Count animals by type
            if class_name in animal_counts:
                animal_counts[class_name] += 1
            else:
                animal_counts[class_name] = 1
            
            # Store detailed information
            animal_details.append({
                "detection_id": i + 1,
                "animal_type": class_name,
                "confidence": float(confidence)
            })
    
    # Save detailed results to JSON (original format for compatibility)
    detailed_results = {
        "image_path": str(image_path),
        "total_animals_detected": num_detections,
        "animal_counts": animal_counts,
        "animal_details": animal_details,
        "confidence_threshold": applied_confidence,
        "timestamp": datetime.now().isoformat()
    }
    
    # Create a subfolder for this image with timestamp
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    image_output_dir = output_dir / f"{image_path.stem}_{timestamp_str}"
    image_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save to YAML in the image-specific folder
    yaml_path = image_output_dir / f"{image_path.stem}_results.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(detailed_results, f, default_flow_style=False, allow_unicode=True)
    
    # Save visualization in the image-specific folder
    vis_path = image_output_dir / f"{image_path.stem}_visualization.jpg"
    if result.boxes is not None:
        # Save image with bounding boxes drawn
        annotated_img = result.plot()
        import cv2
        cv2.imwrite(str(vis_path), annotated_img)
    
    # Return unified format
    return {
        "inference_type": "image",
        "total_animals_detected": num_detections,
        "animal_counts": animal_counts,
        "output_path": str(yaml_path),
        "visualization_path": str(vis_path) if result.boxes is not None else None,
        "confidence_threshold": applied_confidence
    }

def _run_video_inference(
    model,
    video_path: Path,
    output_dir: Path,
    target_fps: int = 15,
    tracker: str = "bt",
    confidence: Optional[float] = None
) -> Dict[str, Any]:
    """Run inference on a video file with animal detection, counting and tracking."""
    #Limitation:
    if not CV2_AVAILABLE:
        raise ImportError("OpenCV is required for video processing. Install with: pip install opencv-python")
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / original_fps if original_fps > 0 else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    tqdm.write("📹 Video properties:")
    tqdm.write(f"   - Original FPS: {original_fps:.2f}")
    tqdm.write(f"   - Total frames: {total_frames}")
    tqdm.write(f"   - Duration: {video_duration:.2f} seconds")
    tqdm.write(f"   - Resolution: {width}x{height}")
    tqdm.write(f"   - Target processing FPS: {target_fps}")
    
    # Calculate frame skip to achieve target FPS
    frame_skip = max(1, int(original_fps / target_fps)) if original_fps > target_fps else 1
    effective_fps = original_fps / frame_skip
    
    tqdm.write(f"   - Frame skip: {frame_skip}")
    tqdm.write(f"   - Effective processing FPS: {effective_fps:.2f}")
    
    # Create video output directory
    video_output_dir = output_dir / f"{video_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    video_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup video output with labels
    output_video_path = video_output_dir / f"{video_path.stem}_labeled.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(str(output_video_path), fourcc, effective_fps, (width, height))

    
    # Configure tracker
    tracker_mapping = {
        "bt": "bytetrack.yaml",
        "bs": "botsort.yaml"
    }
    
    if tracker.lower() not in tracker_mapping:
        raise ValueError(f"Unsupported tracker: {tracker}. Supported trackers: {list(tracker_mapping.keys())}")
    
    tracker_config = tracker_mapping[tracker.lower()]
    tracker_name = "ByteTrack" if tracker.lower() == "bt" else "BoTSORT"
    
    tqdm.write(f"🎯 Using tracker: {tracker_name} ({tracker_config})")
    
    # Determine effective confidence for this run to keep YAML serialization clean
    model_default_conf = getattr(model, "overrides", {}).get("conf")
    if model_default_conf is None:
        model_default_conf = 0.25
    else:
        model_default_conf = float(model_default_conf)

    applied_confidence = float(confidence) if confidence is not None else model_default_conf

    # Initialize tracking variables
    frame_results = []
    frame_count = 0
    processed_frames = 0
    unique_animal_counts = {}  # Track unique animals using tracking
    
    # Process video frames using selected tracker
    tqdm.write(f"🔄 Processing video frames with {tracker_name}...")
    tqdm.write(f"⚡ Skipping {frame_skip - 1} out of every {frame_skip} frames for faster inference")
    
    # Run tracking inference with vid_stride to skip frames during inference (not after)
    track_kwargs = {"conf": applied_confidence} if confidence is not None else {}
    track_kwargs["vid_stride"] = frame_skip  # This makes YOLO skip frames during inference
    
    results = model.track(
        source=str(video_path),
        tracker=tracker_config,
        save=False,
        stream=True,
        verbose=False,
        **track_kwargs
    )
    
    # Calculate expected processed frames
    expected_processed_frames = total_frames // frame_skip
    
    # Create progress bar for video processing (based on frames actually processed)
    pbar = tqdm(total=expected_processed_frames, desc="Processing frames", unit="frame", ncols=100)
    
    for result in results:
        # Get current frame
        frame = result.orig_img
        
        # Update progress bar
        pbar.update(1)
        
        processed_frames += 1
        # Calculate actual frame number in original video (YOLO skips frames for us now)
        frame_count = processed_frames * frame_skip - (frame_skip - 1)
        timestamp = frame_count / original_fps
        
        # Extract detections and process all detected animals
        frame_detections = []
        frame_animal_details = []
        tracked_ids = []
        
        if result.boxes is not None and len(result.boxes) > 0:
            # Get detection information
            boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
            class_ids = result.boxes.cls.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            track_ids = result.boxes.id.cpu().numpy() if result.boxes.id is not None else []
            
            for i, (box, class_id, confidence) in enumerate(zip(boxes, class_ids, confidences)):
                class_name = model.names[int(class_id)]
                
                # Process all detected objects (no filtering)
                track_id = int(track_ids[i]) if i < len(track_ids) else -1
                
                detection = {
                    'animal_type': class_name,
                    'bbox': box.tolist(),  # [x1, y1, x2, y2]
                    'confidence': float(confidence),
                    'detection_id': i + 1,
                    'track_id': track_id
                }
                frame_detections.append(detection)
                
                frame_animal_details.append(detection)
                
                # Track unique animals using track IDs from selected tracker
                if track_id != -1:
                    tracked_ids.append(track_id)
                    # Update unique animal counts
                    if class_name not in unique_animal_counts:
                        unique_animal_counts[class_name] = set()
                    unique_animal_counts[class_name].add(track_id)
        
        # Draw labels on frame
        annotated_frame = draw_detection_labels(frame, frame_detections, show_track_id=True)
        
        # Write frame to output video
        out.write(annotated_frame)
        
        # Store frame results
        frame_result = {
            "frame_number": frame_count,
            "processed_frame": processed_frames,
            "timestamp": timestamp,
            "animals_detected": len(frame_detections),
            "animal_details": frame_animal_details,
            "tracked_animal_ids": tracked_ids  # IDs of tracked animals in this frame
        }
        frame_results.append(frame_result)
    
    # Close progress bar
    pbar.close()
    
    # Release video capture and writer
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # Calculate unique animal counts from selected tracker results
    final_unique_counts = {}
    total_unique_animals = 0
    for animal_type, track_ids in unique_animal_counts.items():
        final_unique_counts[animal_type] = len(track_ids)
        total_unique_animals += len(track_ids)
    
    # Create summary results (without frame-by-frame details)
    video_summary = {
        "video_path": str(video_path),
        "output_video_path": str(output_video_path),
        "video_properties": {
            "original_fps": original_fps,
            "total_frames": total_frames,
            "duration_seconds": video_duration,
            "processed_frames": processed_frames,
            "effective_fps": effective_fps,
            "frame_skip": frame_skip,
            "resolution": f"{width}x{height}"
        },
        "unique_animal_tracking": {
            "total_unique_animals_in_video": total_unique_animals,
            "unique_animal_counts_by_type": final_unique_counts,
            "unique_animal_types": len(final_unique_counts),
            "tracking_method": tracker_name,
            "tracking_parameters": {
                "tracker_config": tracker_config,
                "confidence_threshold": float(applied_confidence)
            }
        },
        "output_directory": str(video_output_dir),
        "timestamp": datetime.now().isoformat()
    }
    
    # Save summary only
    summary_path = video_output_dir / f"{video_path.stem}_video_summary.yaml"
    with open(summary_path, 'w') as f:
        yaml.dump(video_summary, f, default_flow_style=False, allow_unicode=True)

    # Calculate total detections across all frames for unified summary
    total_detections_across_frames = sum(frame_result["animals_detected"] for frame_result in frame_results)

    # Return unified format (similar to image and batch)
    unified_video_results = {
        "inference_type": "video",
        # Use unique individuals as the primary total; include frame-level detections in additional_info
        "total_animals_detected": total_unique_animals,
        # Classification by unique tracked individuals per type
        "animal_counts": final_unique_counts,
        # Point to the summary JSON
        "output_path": str(summary_path),
        "visualization_path": str(output_video_path),
        "confidence_threshold": float(applied_confidence)
    }

    return unified_video_results

def _run_batch_evaluation(
    model,
    data_dir: Path,
    output_dir: Path,
    target_fps: int = 15,
    tracker: str = "bt",
    confidence: Optional[float] = None
) -> Dict[str, Any]:  
    """Run evaluation on a directory of images and/or videos."""
    # Find image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(data_dir.glob(f"**/*{ext}")))
        image_files.extend(list(data_dir.glob(f"**/*{ext.upper()}")))
    
    # Find video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
    video_files = []
    for ext in video_extensions:
        video_files.extend(list(data_dir.glob(f"**/*{ext}")))
        video_files.extend(list(data_dir.glob(f"**/*{ext.upper()}")))
    
    total_files = len(image_files) + len(video_files)
    
    if total_files == 0:
        raise ValueError(f"No images or videos found in {data_dir}")
    
    print(f"📁 Found {total_files} file(s) to process:")
    print(f"   - Images: {len(image_files)}")
    print(f"   - Videos: {len(video_files)}")
    
    # Create batch output directory
    folder_name = data_dir.name  # Use the input folder name
    batch_output_dir = output_dir / f"{folder_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    batch_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process all files
    all_results = []
    total_detections = 0
    combined_animal_counts = {}
    
    # Process images with progress bar
    if image_files:
        print(f"\n📸 Processing {len(image_files)} image(s)...")
        for image_path in tqdm(
            image_files,
            desc="Processing images",
            unit="img",
            ncols=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}"
        ):
            try:
                result = _run_single_image(model, image_path, batch_output_dir, confidence)
                result["file_type"] = "image"
                result["file_name"] = image_path.name
                all_results.append(result)
                total_detections += result["total_animals_detected"]
                
                # Combine animal counts across all files
                for animal_type, count in result["animal_counts"].items():
                    if animal_type in combined_animal_counts:
                        combined_animal_counts[animal_type] += count
                    else:
                        combined_animal_counts[animal_type] = count
                        
            except Exception as e:
                print(f"\n⚠️  Error processing {image_path.name}: {e}")
                continue
    
    # Process videos with progress bar
    if video_files:
        print(f"\n🎬 Processing {len(video_files)} video(s)...")
        for video_path in tqdm(
            video_files,
            desc="Processing videos",
            unit="vid",
            ncols=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}"
        ):
            try:
                result = _run_video_inference(model, video_path, batch_output_dir, target_fps, tracker, confidence)
                result["file_type"] = "video"
                result["file_name"] = video_path.name
                all_results.append(result)
                total_detections += result["total_animals_detected"]
                
                # Combine animal counts across all files
                for animal_type, count in result["animal_counts"].items():
                    if animal_type in combined_animal_counts:
                        combined_animal_counts[animal_type] += count
                    else:
                        combined_animal_counts[animal_type] = count
                        
            except Exception as e:
                print(f"\n⚠️  Error processing {video_path.name}: {e}")
                continue
    
    # Create detailed summary for JSON
    detailed_summary = {
        "data_directory": str(data_dir),
        "total_files_processed": len(all_results),
        "files_by_type": {
            "images": len([r for r in all_results if r.get("file_type") == "image"]),
            "videos": len([r for r in all_results if r.get("file_type") == "video"])
        },
        "total_animals_detected": total_detections,
        "combined_animal_counts": combined_animal_counts,
        "output_directory": str(batch_output_dir),
        "timestamp": datetime.now().isoformat(),
        "individual_results": all_results
    }
    
    # Save detailed summary to YAML
    summary_path = batch_output_dir / "batch_summary.yaml"
    with open(summary_path, 'w') as f:
        yaml.dump(detailed_summary, f, default_flow_style=False, allow_unicode=True)
    
    # Return unified format
    return {
        "inference_type": "batch",
        "total_animals_detected": total_detections,
        "animal_counts": combined_animal_counts,
        "output_path": str(summary_path)
    }

# Example usage and testing
# if __name__ == "__main__":
#     print("🧪 EVALUATION MODULE TEST")
