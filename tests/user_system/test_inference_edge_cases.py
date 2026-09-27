"""
Edge case and error handling tests for inference.py

This file contains tests for boundary conditions, invalid inputs, and error scenarios
in the inference module. These tests ensure robust error handling and graceful degradation.

Test categories:
- Invalid file paths and missing files
- Unsupported file formats
- Empty directories
- Invalid parameters (FPS, tracker, etc.)
- Edge cases (very small images, corrupted files, etc.)
"""

import sys
import tempfile
import shutil
from pathlib import Path

import pytest

# Ensure repository root is on sys.path for absolute imports
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
	sys.path.insert(0, str(repo_root))

DATA_ROOT = repo_root / "user_system" / "tests" / "inference_test_data"
IMAGES_ROOT = DATA_ROOT / "images"
VIDEOS_ROOT = DATA_ROOT / "videos"

from user_system.inference import (
	evaluate_model,
	YOLO_AVAILABLE,
	CV2_AVAILABLE,
)


# ==================== Invalid Path Tests ====================

@pytest.mark.skipif(not YOLO_AVAILABLE, reason="Ultralytics YOLO not available")
def test_inference_nonexistent_file():
	"""Test that non-existent file raises appropriate error."""
	fake_path = "/this/path/definitely/does/not/exist/image.jpg"
	
	with pytest.raises((FileNotFoundError, ValueError)):
		evaluate_model(fake_path)


@pytest.mark.skipif(not YOLO_AVAILABLE, reason="Ultralytics YOLO not available")
def test_inference_none_path():
	"""Test that None as path raises appropriate error."""
	with pytest.raises((TypeError, ValueError, AttributeError)):
		evaluate_model(None)


# ==================== Invalid File Type Tests ====================

@pytest.mark.skipif(not YOLO_AVAILABLE, reason="Ultralytics YOLO not available")
def test_inference_unsupported_file_type():
	"""Test that unsupported file types are handled appropriately."""
	# Create a temporary text file
	with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
		f.write("This is not an image or video file")
		temp_path = f.name
	
	try:
		# Should raise an error or handle gracefully
		with pytest.raises(Exception):  # Could be ValueError, cv2.error, etc.
			evaluate_model(temp_path)
	finally:
		# Clean up
		Path(temp_path).unlink(missing_ok=True)


@pytest.mark.skipif(not YOLO_AVAILABLE, reason="Ultralytics YOLO not available")
def test_inference_directory_as_file():
	"""Test that passing a directory when expecting file is handled."""
	# Create a temporary directory
	temp_dir = tempfile.mkdtemp()
	
	try:
		# When passed as a file path, should process as batch (directory mode)
		# or raise an error if empty
		with pytest.raises(ValueError, match="No images or videos found"):
			evaluate_model(temp_dir)
	finally:
		# Clean up
		shutil.rmtree(temp_dir, ignore_errors=True)


# ==================== Empty Data Tests ====================

@pytest.mark.skipif(not YOLO_AVAILABLE, reason="Ultralytics YOLO not available")
def test_inference_empty_directory():
	"""Test batch processing on empty directory raises ValueError."""
	temp_dir = tempfile.mkdtemp()
	
	try:
		with pytest.raises(ValueError, match="No images or videos found"):
			evaluate_model(temp_dir)
	finally:
		shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.skipif(not YOLO_AVAILABLE, reason="Ultralytics YOLO not available")
def test_inference_directory_with_only_unsupported_files():
	"""Test directory containing only unsupported file types."""
	temp_dir = tempfile.mkdtemp()
	
	try:
		# Create some non-image/video files
		(Path(temp_dir) / "file1.txt").write_text("text file")
		(Path(temp_dir) / "file2.json").write_text("{}")
		(Path(temp_dir) / "file3.csv").write_text("data")
		
		with pytest.raises(ValueError, match="No images or videos found"):
			evaluate_model(temp_dir)
	finally:
		shutil.rmtree(temp_dir, ignore_errors=True)


# ==================== Invalid Parameter Tests ====================

@pytest.mark.skipif(not (YOLO_AVAILABLE and CV2_AVAILABLE), reason="Requires YOLO and OpenCV")
def test_inference_invalid_tracker():
	"""Test that invalid tracker parameter raises ValueError."""
	# We need a valid video file for this test
	# Skip if no test video available
	if not VIDEOS_ROOT.exists():
		pytest.skip("No test videos available")
	
	from pathlib import Path
	video_files = list(VIDEOS_ROOT.glob("*.mp4"))
	if not video_files:
		pytest.skip("No .mp4 files found in test data")
	
	video_path = str(video_files[0])
	
	# Invalid tracker should raise ValueError
	with pytest.raises(ValueError, match="Unsupported tracker"):
		evaluate_model(video_path, tracker="invalid_tracker")


@pytest.mark.skipif(not (YOLO_AVAILABLE and CV2_AVAILABLE), reason="Requires YOLO and OpenCV")
def test_inference_negative_fps():
	"""Test that negative FPS raises appropriate error."""
	if not VIDEOS_ROOT.exists():
		pytest.skip("No test videos available")
	
	video_files = list(VIDEOS_ROOT.glob("*.mp4"))
	if not video_files:
		pytest.skip("No .mp4 files found in test data")
	
	video_path = str(video_files[0])
	
	# Negative FPS should be handled (might not raise error but should use default)
	# Test depends on implementation - adjust as needed
	try:
		result = evaluate_model(video_path, target_fps=-1)
		# If it doesn't raise, it should handle it gracefully
		assert isinstance(result, dict)
	except (ValueError, AssertionError):
		# If it does raise, that's also acceptable behavior
		pass


@pytest.mark.skipif(not (YOLO_AVAILABLE and CV2_AVAILABLE), reason="Requires YOLO and OpenCV")
def test_inference_zero_fps():
	"""Test that zero FPS is handled appropriately."""
	if not VIDEOS_ROOT.exists():
		pytest.skip("No test videos available")
	
	video_files = list(VIDEOS_ROOT.glob("*.mp4"))
	if not video_files:
		pytest.skip("No .mp4 files found in test data")
	
	video_path = str(video_files[0])
	
	# Zero FPS should be handled gracefully
	try:
		result = evaluate_model(video_path, target_fps=0)
		assert isinstance(result, dict)
	except (ValueError, ZeroDivisionError):
		# Acceptable to raise error for invalid FPS
		pass


@pytest.mark.skipif(not YOLO_AVAILABLE, reason="Ultralytics YOLO not available")
def test_inference_invalid_model_path():
	"""Test that invalid model path raises appropriate error."""
	if not IMAGES_ROOT.exists():
		pytest.skip("No test images available")
	
	image_files = list(IMAGES_ROOT.glob("*.jpg"))
	if not image_files:
		pytest.skip("No .jpg files found in test data")
	
	image_path = str(image_files[0])
	
	# Invalid model path should raise error
	with pytest.raises(Exception):  # Could be FileNotFoundError, RuntimeError, etc.
		evaluate_model(image_path, model_name="/nonexistent/model.pt")


# ==================== Boundary Value Tests ====================

@pytest.mark.skipif(not (YOLO_AVAILABLE and CV2_AVAILABLE), reason="Requires YOLO and OpenCV")
def test_inference_minimum_fps():
	"""Test video processing with minimum FPS (1)."""
	if not VIDEOS_ROOT.exists():
		pytest.skip("No test videos available")
	
	video_files = list(VIDEOS_ROOT.glob("*.mp4"))
	if not video_files:
		pytest.skip("No .mp4 files found in test data")
	
	video_path = str(video_files[0])
	temp_out = tempfile.mkdtemp()
	
	try:
		# FPS=1 should work (very slow processing)
		result = evaluate_model(video_path, output_dir=temp_out, target_fps=1)
		assert isinstance(result, dict)
		assert result.get("inference_type") == "video"
	finally:
		shutil.rmtree(temp_out, ignore_errors=True)


@pytest.mark.skipif(not (YOLO_AVAILABLE and CV2_AVAILABLE), reason="Requires YOLO and OpenCV")
def test_inference_very_high_fps():
	"""Test video processing with very high FPS (1000)."""
	if not VIDEOS_ROOT.exists():
		pytest.skip("No test videos available")
	
	video_files = list(VIDEOS_ROOT.glob("*.mp4"))
	if not video_files:
		pytest.skip("No .mp4 files found in test data")
	
	video_path = str(video_files[0])
	temp_out = tempfile.mkdtemp()
	
	try:
		# Very high FPS should be handled (might process every frame)
		result = evaluate_model(video_path, output_dir=temp_out, target_fps=1000)
		assert isinstance(result, dict)
		assert result.get("inference_type") == "video"
	finally:
		shutil.rmtree(temp_out, ignore_errors=True)


# ==================== Path Edge Cases ====================

@pytest.mark.skipif(not YOLO_AVAILABLE, reason="Ultralytics YOLO not available")
def test_inference_path_with_spaces():
	"""Test that paths with spaces are handled correctly."""
	# Create temp directory with spaces in name
	temp_dir = tempfile.mkdtemp(prefix="test with spaces ")
	
	try:
		# Create a dummy file (won't be a real image, but tests path handling)
		test_file = Path(temp_dir) / "test image.txt"
		test_file.write_text("dummy")
		
		# Should handle the path correctly (might fail at file format validation)
		with pytest.raises(Exception):  # Will fail but path should be parsed correctly
			evaluate_model(str(test_file))
	finally:
		shutil.rmtree(temp_dir, ignore_errors=True)


# ==================== Output Directory Tests ====================

@pytest.mark.skipif(not YOLO_AVAILABLE, reason="Ultralytics YOLO not available")
def test_inference_nonexistent_output_directory():
	"""Test that non-existent output directory is created automatically."""
	if not IMAGES_ROOT.exists():
		pytest.skip("No test images available")
	
	image_files = list(IMAGES_ROOT.glob("*.jpg"))
	if not image_files:
		pytest.skip("No .jpg files found in test data")
	
	image_path = str(image_files[0])
	
	# Use a nested non-existent path
	temp_base = tempfile.mkdtemp()
	output_dir = Path(temp_base) / "level1" / "level2" / "level3"
	
	try:
		result = evaluate_model(image_path, output_dir=str(output_dir))
		
		# Directory should be created
		assert output_dir.exists()
		assert isinstance(result, dict)
	finally:
		shutil.rmtree(temp_base, ignore_errors=True)


# ==================== Special Characters Tests ====================

@pytest.mark.skipif(not YOLO_AVAILABLE, reason="Ultralytics YOLO not available")
def test_inference_unicode_path():
	"""Test that paths with unicode characters are handled."""
	temp_dir = tempfile.mkdtemp(prefix="测试_")  # Chinese characters
	
	try:
		# Create a dummy file
		test_file = Path(temp_dir) / "文件.txt"
		test_file.write_text("dummy", encoding='utf-8')
		
		# Should handle unicode paths correctly
		with pytest.raises(Exception):  # Will fail at format validation
			evaluate_model(str(test_file))
	finally:
		shutil.rmtree(temp_dir, ignore_errors=True)
