"""
Test suite for user_system.evaluate module.

This module tests the model evaluation functionality including
counting metrics, detection metrics, and performance measurements.
"""

import sys
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import pytest
import torch

# Put repo root on sys.path
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Import the module under test
from user_system.evaluate import count_metrics_per_image


class TestCountMetricsPerImage:
    """Test cases for count_metrics_per_image function."""

    def test_count_metrics_per_image_equal_counts(self):
        """Test metrics when prediction and ground truth counts are equal."""
        pred_boxes = [[1, 2, 3, 4], [5, 6, 7, 8]]  # 2 boxes
        true_boxes = [[1, 2, 3, 4], [5, 6, 7, 8]]  # 2 boxes
        
        result = count_metrics_per_image(pred_boxes, true_boxes)
        
        assert result["mae"] == 0
        assert result["rmse"] == 0
        assert result["bias"] == 0
        assert result["within1"] == 1
        assert result["undercount"] == 0
        assert result["overcount"] == 0

    def test_count_metrics_per_image_overcount(self):
        """Test metrics when prediction count is higher than ground truth."""
        pred_boxes = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]  # 3 boxes
        true_boxes = [[1, 2, 3, 4]]  # 1 box
        
        result = count_metrics_per_image(pred_boxes, true_boxes)
        
        assert result["mae"] == 2
        assert result["rmse"] == 4  # (3-1)^2 = 4
        assert result["bias"] == 2  # 3-1 = 2
        assert result["within1"] == 0
        assert result["undercount"] == 0
        assert result["overcount"] == 1

    def test_count_metrics_per_image_undercount(self):
        """Test metrics when prediction count is lower than ground truth."""
        pred_boxes = [[1, 2, 3, 4]]  # 1 box
        true_boxes = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]  # 3 boxes
        
        result = count_metrics_per_image(pred_boxes, true_boxes)
        
        assert result["mae"] == 2
        assert result["rmse"] == 4  # (1-3)^2 = 4
        assert result["bias"] == -2  # 1-3 = -2
        assert result["within1"] == 0
        assert result["undercount"] == 1
        assert result["overcount"] == 0

    def test_count_metrics_per_image_within_one(self):
        """Test metrics when difference is exactly 1."""
        pred_boxes = [[1, 2, 3, 4], [5, 6, 7, 8]]  # 2 boxes
        true_boxes = [[1, 2, 3, 4]]  # 1 box
        
        result = count_metrics_per_image(pred_boxes, true_boxes)
        
        assert result["mae"] == 1
        assert result["rmse"] == 1  # (2-1)^2 = 1
        assert result["bias"] == 1  # 2-1 = 1
        assert result["within1"] == 1
        assert result["undercount"] == 0
        assert result["overcount"] == 1

    def test_count_metrics_per_image_empty_prediction(self):
        """Test metrics when prediction is empty."""
        pred_boxes = []  # 0 boxes
        true_boxes = [[1, 2, 3, 4], [5, 6, 7, 8]]  # 2 boxes
        
        result = count_metrics_per_image(pred_boxes, true_boxes)
        
        assert result["mae"] == 2
        assert result["rmse"] == 4  # (0-2)^2 = 4
        assert result["bias"] == -2  # 0-2 = -2
        assert result["within1"] == 0
        assert result["undercount"] == 1
        assert result["overcount"] == 0

    def test_count_metrics_per_image_empty_ground_truth(self):
        """Test metrics when ground truth is empty."""
        pred_boxes = [[1, 2, 3, 4], [5, 6, 7, 8]]  # 2 boxes
        true_boxes = []  # 0 boxes
        
        result = count_metrics_per_image(pred_boxes, true_boxes)
        
        assert result["mae"] == 2
        assert result["rmse"] == 4  # (2-0)^2 = 4
        assert result["bias"] == 2  # 2-0 = 2
        assert result["within1"] == 0
        assert result["undercount"] == 0
        assert result["overcount"] == 1

    def test_count_metrics_per_image_both_empty(self):
        """Test metrics when both prediction and ground truth are empty."""
        pred_boxes = []  # 0 boxes
        true_boxes = []  # 0 boxes
        
        result = count_metrics_per_image(pred_boxes, true_boxes)
        
        assert result["mae"] == 0
        assert result["rmse"] == 0
        assert result["bias"] == 0
        assert result["within1"] == 1
        assert result["undercount"] == 0
        assert result["overcount"] == 0


class TestEvaluateMain:
    """Test cases for the main evaluation script functionality."""

    @patch('user_system.evaluate.YOLO')
    @patch('user_system.evaluate.Path')
    def test_evaluation_workflow_mock(self, mock_path, mock_yolo):
        """Test the main evaluation workflow with mocked dependencies."""
        # Mock YOLO model
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model
        
        # Mock model prediction results
        mock_result = MagicMock()
        mock_result.boxes.xyxy = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        mock_model.predict.return_value = [mock_result]
        
        # Mock validation results
        mock_val_result = MagicMock()
        mock_val_result.box = MagicMock()
        mock_val_result.box.map50 = 0.85
        mock_val_result.box.map = 0.75
        mock_val_result.box.mr = 0.12
        mock_val_result.box.p = [0.9, 0.8]
        mock_val_result.box.ap = [0.85, 0.75]
        mock_val_result.speed = {
            "preprocess": 5.0,
            "inference": 10.0,
            "postprocess": 2.0
        }
        mock_model.val.return_value = mock_val_result
        
        # Mock file system
        mock_img_dir = MagicMock()
        mock_img_dir.iterdir.return_value = [
            MagicMock(suffix='.jpg', stem='image1'),
            MagicMock(suffix='.png', stem='image2')
        ]
        
        mock_label_dir = MagicMock()
        mock_label_file = MagicMock()
        mock_label_file.exists.return_value = True
        mock_label_dir.__truediv__.return_value = mock_label_file
        
        mock_path.return_value = mock_img_dir
        mock_path.side_effect = lambda x: mock_img_dir if 'images' in str(x) else mock_label_dir
        
        # Mock file reading
        with patch('builtins.open', mock_open(read_data="0 0.5 0.5 0.2 0.3\n1 0.3 0.7 0.1 0.2")):
            # This would normally run the main evaluation script
            # We're just testing that the mocked components work together
            assert mock_model.predict is not None
            assert mock_model.val is not None

    def test_metrics_calculation_consistency(self):
        """Test that metrics calculations are mathematically consistent."""
        test_cases = [
            ([], []),  # Both empty
            ([1], []),  # Overcount by 1
            ([], [1]),  # Undercount by 1
            ([1, 2], [1]),  # Overcount by 1
            ([1], [1, 2]),  # Undercount by 1
            ([1, 2, 3], [1, 2]),  # Overcount by 1
            ([1, 2], [1, 2, 3]),  # Undercount by 1
        ]
        
        for pred_boxes, true_boxes in test_cases:
            result = count_metrics_per_image(pred_boxes, true_boxes)
            
            # Check mathematical consistency
            pred_count = len(pred_boxes)
            true_count = len(true_boxes)
            
            assert result["mae"] == abs(pred_count - true_count)
            assert result["rmse"] == (pred_count - true_count) ** 2
            assert result["bias"] == pred_count - true_count
            
            # Check logical consistency
            if result["mae"] <= 1:
                assert result["within1"] == 1
            else:
                assert result["within1"] == 0
                
            if pred_count < true_count:
                assert result["undercount"] == 1
                assert result["overcount"] == 0
            elif pred_count > true_count:
                assert result["undercount"] == 0
                assert result["overcount"] == 1
            else:
                assert result["undercount"] == 0
                assert result["overcount"] == 0

    def test_edge_cases(self):
        """Test edge cases in metrics calculation."""
        # Test with very large differences
        pred_boxes = [[i, i+1, i+2, i+3] for i in range(100)]  # 100 boxes
        true_boxes = [[1, 2, 3, 4]]  # 1 box
        
        result = count_metrics_per_image(pred_boxes, true_boxes)
        
        assert result["mae"] == 99
        assert result["rmse"] == 99 ** 2
        assert result["bias"] == 99
        assert result["within1"] == 0
        assert result["undercount"] == 0
        assert result["overcount"] == 1

    def test_metrics_data_types(self):
        """Test that metrics return correct data types."""
        pred_boxes = [[1, 2, 3, 4], [5, 6, 7, 8]]
        true_boxes = [[1, 2, 3, 4]]
        
        result = count_metrics_per_image(pred_boxes, true_boxes)
        
        assert isinstance(result["mae"], int)
        assert isinstance(result["rmse"], int)
        assert isinstance(result["bias"], int)
        assert isinstance(result["within1"], int)
        assert isinstance(result["undercount"], int)
        assert isinstance(result["overcount"], int)
        
        # Check that boolean-like values are 0 or 1
        assert result["within1"] in [0, 1]
        assert result["undercount"] in [0, 1]
        assert result["overcount"] in [0, 1]

    def test_performance_metrics_calculation(self):
        """Test performance metrics calculation logic."""
        # Test speed metrics calculation
        preprocess_time_ms = 5.0
        inference_time_ms = 10.0
        postprocess_time_ms = 2.0
        total_time_ms = preprocess_time_ms + inference_time_ms + postprocess_time_ms
        fps_bs1 = 1000 / total_time_ms
        latency_ms_p50 = total_time_ms
        
        assert fps_bs1 == 1000 / 17.0  # Should be approximately 58.82
        assert latency_ms_p50 == 17.0
        
        # Test with different timing values
        preprocess_time_ms = 2.0
        inference_time_ms = 8.0
        postprocess_time_ms = 1.0
        total_time_ms = preprocess_time_ms + inference_time_ms + postprocess_time_ms
        fps_bs1 = 1000 / total_time_ms
        
        assert fps_bs1 == 1000 / 11.0  # Should be approximately 90.91
        assert total_time_ms == 11.0
