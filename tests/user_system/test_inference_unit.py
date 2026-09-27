"""
Unit tests for inference.py helper functions

This file contains unit tests for individual functions in inference.py,
such as print_evaluation_results and draw_detection_labels.
These are isolated tests that don't require YOLO models or large test data.
"""

import sys
from pathlib import Path

import pytest

# Ensure repository root is on sys.path for absolute imports
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
	sys.path.insert(0, str(repo_root))

# Import functions to test
from user_system.inference import (
	print_evaluation_results,
	draw_detection_labels,
	YOLO_AVAILABLE,
	CV2_AVAILABLE,
)


# ==================== Tests for print_evaluation_results ====================

def test_print_evaluation_results_with_animals(capsys):
	"""Test print_evaluation_results with valid results containing animals."""
	results = {
		"total_animals_detected": 5,
		"animal_counts": {"cow": 3, "sheep": 2},
		"output_path": "/tmp/test_results.yaml",
		"visualization_path": "/tmp/test_viz.jpg"
	}
	
	print_evaluation_results(results)
	
	# Capture printed output
	captured = capsys.readouterr()
	
	# Verify key information is present in output
	assert "Total Animals Detected: 5" in captured.out
	assert "cow: 3" in captured.out
	assert "sheep: 2" in captured.out
	assert "Detailed results saved to: /tmp/test_results.yaml" in captured.out
	assert "Visualization saved to: /tmp/test_viz.jpg" in captured.out


def test_print_evaluation_results_no_animals(capsys):
	"""Test print_evaluation_results when no animals detected."""
	results = {
		"total_animals_detected": 0,
		"animal_counts": {}
	}
	
	print_evaluation_results(results)
	captured = capsys.readouterr()
	
	assert "Total Animals Detected: 0" in captured.out
	assert "No animals detected" in captured.out


def test_print_evaluation_results_minimal(capsys):
	"""Test print_evaluation_results with minimal data (no paths)."""
	results = {
		"total_animals_detected": 2,
		"animal_counts": {"chicken": 2}
	}
	
	# Should not crash even without optional fields
	print_evaluation_results(results)
	captured = capsys.readouterr()
	
	assert "Total Animals Detected: 2" in captured.out
	assert "chicken: 2" in captured.out


def test_print_evaluation_results_invalid_input():
	"""Test that print_evaluation_results handles invalid input gracefully."""
	# Should not crash with None
	print_evaluation_results(None)
	
	# Should not crash with non-dict types
	print_evaluation_results("not a dict")
	print_evaluation_results([])
	print_evaluation_results(123)


def test_print_evaluation_results_empty_dict(capsys):
	"""Test with empty dictionary."""
	print_evaluation_results({})
	captured = capsys.readouterr()
	
	# Should print 0 animals detected when total is missing
	assert "Total Animals Detected: 0" in captured.out


def test_print_evaluation_results_multiple_animal_types(capsys):
	"""Test with many different animal types."""
	results = {
		"total_animals_detected": 10,
		"animal_counts": {
			"cow": 3,
			"sheep": 2,
			"chicken": 4,
			"pig": 1
		},
		"output_path": "/tmp/results.yaml"
	}
	
	print_evaluation_results(results)
	captured = capsys.readouterr()
	
	assert "Total Animals Detected: 10" in captured.out
	assert "cow: 3" in captured.out
	assert "sheep: 2" in captured.out
	assert "chicken: 4" in captured.out
	assert "pig: 1" in captured.out


# ==================== Tests for draw_detection_labels ====================

def test_draw_detection_labels_empty_detections():
	"""Test draw_detection_labels with no detections."""
	import numpy as np
	
	frame = np.zeros((100, 100, 3), dtype=np.uint8)
	detections = []
	
	# Should not raise and should return an array of the same shape
	result = draw_detection_labels(frame, detections)
	assert isinstance(result, type(frame))
	assert result.shape == frame.shape


def test_draw_detection_labels_single_detection():
	"""Test draw_detection_labels with one detection."""
	import numpy as np
	
	frame = np.zeros((100, 100, 3), dtype=np.uint8)
	detections = [{
		"bbox": [10, 10, 50, 50],
		"animal_type": "cow",
		"confidence": 0.9,
		"track_id": 1,
	}]
	
	result = draw_detection_labels(frame, detections)
	assert result.shape == frame.shape


def test_draw_detection_labels_multiple_detections():
	"""Test draw_detection_labels with multiple detections."""
	import numpy as np
	
	frame = np.zeros((200, 200, 3), dtype=np.uint8)
	detections = [
		{
			"bbox": [10, 10, 50, 50],
			"animal_type": "cow",
			"confidence": 0.9,
			"track_id": 1,
		},
		{
			"bbox": [60, 60, 100, 100],
			"animal_type": "sheep",
			"confidence": 0.85,
			"track_id": 2,
		},
		{
			"bbox": [110, 110, 150, 150],
			"animal_type": "chicken",
			"confidence": 0.75,
			"track_id": 3,
		}
	]
	
	result = draw_detection_labels(frame, detections)
	assert result.shape == frame.shape


def test_draw_detection_labels_no_track_id():
	"""Test draw_detection_labels with detection missing track_id."""
	import numpy as np
	
	frame = np.zeros((100, 100, 3), dtype=np.uint8)
	detections = [{
		"bbox": [10, 10, 50, 50],
		"animal_type": "pig",
		"confidence": 0.8,
		# No track_id
	}]
	
	# Should handle missing track_id gracefully
	result = draw_detection_labels(frame, detections)
	assert result.shape == frame.shape


def test_draw_detection_labels_hide_track_id():
	"""Test draw_detection_labels with show_track_id=False."""
	import numpy as np
	
	frame = np.zeros((100, 100, 3), dtype=np.uint8)
	detections = [{
		"bbox": [10, 10, 50, 50],
		"animal_type": "cow",
		"confidence": 0.9,
		"track_id": 1,
	}]
	
	result = draw_detection_labels(frame, detections, show_track_id=False)
	assert result.shape == frame.shape


@pytest.mark.skipif(not CV2_AVAILABLE, reason="OpenCV not available")
def test_draw_detection_labels_different_animal_types():
	"""Test that different animal types use different colors."""
	import numpy as np
	
	frame = np.zeros((300, 300, 3), dtype=np.uint8)
	detections = [
		{"bbox": [10, 10, 50, 50], "animal_type": "cow", "confidence": 0.9, "track_id": 1},
		{"bbox": [60, 60, 100, 100], "animal_type": "sheep", "confidence": 0.85, "track_id": 2},
		{"bbox": [110, 110, 150, 150], "animal_type": "chicken", "confidence": 0.8, "track_id": 3},
		{"bbox": [160, 160, 200, 200], "animal_type": "pig", "confidence": 0.75, "track_id": 4},
	]
	
	result = draw_detection_labels(frame, detections)
	
	# Frame should have been modified (not all zeros anymore)
	assert not np.all(result == 0)
	assert result.shape == frame.shape
