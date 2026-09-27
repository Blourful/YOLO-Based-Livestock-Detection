"""
Test suite for user_system.main module.

This module tests the main CLI functionality including
argument parsing, mode execution, and workflow functions.
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import pytest
import argparse

# Put repo root on sys.path
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Shim for utils modules to avoid import errors
def _ensure_utils_shim():
    """Ensure utils modules are available in sys.modules."""
    if 'utils' not in sys.modules:
        utils_mock = MagicMock()
        utils_mock.arg_helper = MagicMock()
        utils_mock.ui_helper = MagicMock()
        utils_mock.directory_helper = MagicMock()
        utils_mock.modify_helper = MagicMock()
        utils_mock.hardware_info = MagicMock()
        sys.modules['utils'] = utils_mock
        sys.modules['utils.arg_helper'] = utils_mock.arg_helper
        sys.modules['utils.ui_helper'] = utils_mock.ui_helper
        sys.modules['utils.directory_helper'] = utils_mock.directory_helper
        sys.modules['utils.modify_helper'] = utils_mock.modify_helper
        sys.modules['utils.hardware_info'] = utils_mock.hardware_info
    
    # Add other required modules
    if 'augment' not in sys.modules:
        augment_mock = MagicMock()
        sys.modules['augment'] = augment_mock
    
    if 'inference' not in sys.modules:
        inference_mock = MagicMock()
        sys.modules['inference'] = inference_mock
    
    if 'train' not in sys.modules:
        train_mock = MagicMock()
        sys.modules['train'] = train_mock

# Apply the shim
_ensure_utils_shim()

# Import the module under test
from user_system.main import (
    build_parser,
    run_compile,
    run_train,
    run_eval,
    run_augment,
    run_labels,
    run_recommend,
    main
)


class TestBuildParser:
    """Test cases for build_parser function."""

    def test_build_parser_creates_parser(self):
        """Test that build_parser creates a valid ArgumentParser."""
        parser = build_parser()
        assert isinstance(parser, argparse.ArgumentParser)
        assert parser.description == "YOLO Livestock Detection CLI with Automatic Dataset Compilation"

    def test_build_parser_has_subparsers(self):
        """Test that build_parser has subparsers for different modes."""
        parser = build_parser()
        subparsers = parser._subparsers
        assert subparsers is not None
        
        # Check that required modes are available
        actions = subparsers._actions
        mode_choices = set()
        for action in actions:
            if hasattr(action, 'choices') and action.choices:
                mode_choices.update(action.choices.keys())
        
        expected_modes = {'compile', 'train', 'eval', 'labels', 'augment', 'help', 'recommend'}
        assert expected_modes.issubset(mode_choices)

    def test_build_parser_compile_subparser(self):
        """Test compile subparser arguments."""
        parser = build_parser()
        
        # Test compile mode
        args = parser.parse_args(['compile', '--compile-inputs', 'dataset1', 'dataset2', '--data', 'output'])
        assert args.mode == 'compile'
        assert args.compile_inputs == ['dataset1', 'dataset2']
        assert args.data == 'output'

    def test_build_parser_train_subparser(self):
        """Test train subparser arguments."""
        parser = build_parser()
        
        # Test train mode
        args = parser.parse_args(['train', '--data', 'dataset', '--epochs', '50'])
        assert args.mode == 'train'
        assert args.data == 'dataset'
        assert args.epochs == 50

    def test_build_parser_eval_subparser(self):
        """Test eval subparser arguments."""
        parser = build_parser()
        
        # Test eval mode
        args = parser.parse_args(['eval', '--data', 'test_data', '--model', 'model.pt'])
        assert args.mode == 'eval'
        assert args.data == 'test_data'
        assert args.model == 'model.pt'

    def test_build_parser_labels_subparser(self):
        """Test labels subparser arguments."""
        parser = build_parser()
        
        # Test labels mode
        args = parser.parse_args(['labels', '--data', 'dataset', '--old-label', 'cow', '--new-label', 'cattle'])
        assert args.mode == 'labels'
        assert args.data == 'dataset'
        assert args.old_label == 'cow'
        assert args.new_label == 'cattle'

    def test_build_parser_augment_subparser(self):
        """Test augment subparser arguments."""
        parser = build_parser()
        
        # Test augment mode
        args = parser.parse_args(['augment', '--inputs', 'dataset1', 'dataset2', '--workers', '4'])
        assert args.mode == 'augment'
        assert args.inputs == ['dataset1', 'dataset2']
        assert args.workers == 4

    def test_build_parser_help_subparser(self):
        """Test help subparser."""
        parser = build_parser()
        
        # Test help mode
        args = parser.parse_args(['help'])
        assert args.mode == 'help'

    def test_build_parser_recommend_subparser(self):
        """Test recommend subparser."""
        parser = build_parser()
        
        # Test recommend mode
        args = parser.parse_args(['recommend'])
        assert args.mode == 'recommend'


class TestRunCompile:
    """Test cases for run_compile function."""

    def test_run_compile_success(self, capsys):
        """Test successful compilation."""
        args = MagicMock()
        args.compile_inputs = ["./dataset1", "./dataset2"]
        args.compile_out = "./output"

        with patch('user_system.main.validate_compile_args', return_value=True), \
             patch('user_system.main.prepare_dataset', return_value=Path("./output")):        
            result = run_compile(args)

        assert result is None
        captured = capsys.readouterr()
        assert "COMPILE MODE ACTIVATED" in captured.out
        assert "Compilation complete" in captured.out

    def test_run_compile_validation_failure(self, capsys):
        """Test compilation with validation failure."""
        args = MagicMock()
        
        with patch('user_system.main.validate_compile_args', return_value=False):
            result = run_compile(args)
        
        assert result == 0
        captured = capsys.readouterr()
        assert "COMPILE MODE ACTIVATED" in captured.out


class TestRunTrain:
    """Test cases for run_train function."""

    def test_run_train_success(self, capsys):
        """Test successful training."""
        args = MagicMock()
        args.data = "./dataset"
        args.epochs = 50
        args.batch = 16
        args.imgsz = 640
        args.preset = None
        args.output = "./runs/train"
        args.lr = None
        args.optimizer = None
        args.device = None
        args._cli_flags = set()
        
        with patch('user_system.main.validate_train_args', return_value=True), \
             patch('user_system.main.prepare_dataset', return_value=Path("./dataset")), \
             patch('user_system.main.train_from_args', return_value=Path("./runs/train/exp")):
            run_train(args)
        
        captured = capsys.readouterr()
        assert "TRAINING MODE ACTIVATED" in captured.out
        assert "TRAINING PHASE" in captured.out
        assert "Training finished" in captured.out

    def test_run_train_validation_failure(self, capsys):
        """Test training with validation failure."""
        args = MagicMock()
        
        with patch('user_system.main.validate_train_args', return_value=False):
            run_train(args)
        
        captured = capsys.readouterr()
        assert "TRAINING MODE ACTIVATED" in captured.out

    def test_run_train_with_batch_size_alias(self, capsys):
        """Test training with batch_size alias."""
        args = MagicMock()
        args.data = "./dataset"
        args.epochs = 50
        args.batch = None
        args.batch_size = 32
        args.imgsz = 640
        args.preset = None
        args.output = "./runs/train"
        args.lr = None
        args.optimizer = None
        args.device = None
        args._cli_flags = set()
        
        with patch('user_system.main.validate_train_args', return_value=True), \
             patch('user_system.main.prepare_dataset', return_value=Path("./dataset")), \
             patch('user_system.main.train_from_args', return_value=Path("./runs/train/exp")):
            run_train(args)
        
        # Check that batch_size was copied to batch
        assert args.batch == 32


class TestRunEval:
    """Test cases for run_eval function."""

    def test_run_eval_success(self, capsys):
        """Test successful evaluation."""
        args = MagicMock()
        args.data = "./test_data"
        args.output = "./runs/eval"
        args.model = "model.pt"
        args.fps = 15
        args.tracker = "bt"
        
        with patch('user_system.main.validate_eval_args', return_value=Path("./test_data")), \
             patch('user_system.main.evaluate_model') as mock_eval:
            run_eval(args)
        
        captured = capsys.readouterr()
        assert "INFERENCE MODE ACTIVATED" in captured.out
        
        mock_eval.assert_called_once_with(
            data_path='test_data',
            output_dir="./runs/eval",
            model_name="model.pt",
            target_fps=15,
            tracker="bt"
        )

    def test_run_eval_validation_failure(self, capsys):
        """Test evaluation with validation failure."""
        args = MagicMock()
        
        with patch('user_system.main.validate_eval_args', return_value=None):
            run_eval(args)
        
        captured = capsys.readouterr()
        assert "INFERENCE MODE ACTIVATED" in captured.out

    def test_run_eval_exception_handling(self, capsys):
        """Test evaluation with exception handling."""
        args = MagicMock()
        args.data = "./test_data"
        args.output = "./runs/eval"
        args.model = "model.pt"
        args.fps = 15
        args.tracker = "bt"
        
        with patch('user_system.main.validate_eval_args', return_value=Path("./test_data")), \
             patch('user_system.main.evaluate_model', side_effect=Exception("Test error")):
            run_eval(args)
        
        captured = capsys.readouterr()
        assert "Error: Test error" in captured.out
        assert "Troubleshooting tips" in captured.out


class TestRunAugment:
    """Test cases for run_augment function."""

    def test_run_augment_success(self, capsys):
        """Test successful augmentation."""
        args = MagicMock()
        args.inputs = ["./dataset1", "./dataset2"]
        args.workers = 4
        
        mock_data_paths = [Path("./dataset1"), Path("./dataset2")]
        mock_config_path = Path("./config.yaml")
        mock_config = {"modes": {"flip_h": {"enabled": True}}}
        
        with patch('user_system.main.validate_augment_args', return_value=(mock_data_paths, mock_config_path, mock_config)), \
             patch('user_system.main.get_base_directory', return_value=Path("./base")), \
             patch('user_system.main.setup_augmentation_environment', return_value=[Path("./output1"), Path("./output2")]), \
             patch('user_system.main.run_augmentation_pipeline', return_value={'total_processed': 100, 'total_created': 200, 'expansion_ratio': 2.0}):
            run_augment(args)
        
        captured = capsys.readouterr()
        assert "AUGMENTATION MODE ACTIVATED" in captured.out
        assert "Input validation passed" in captured.out
        assert "AUGMENTATION COMPLETE" in captured.out
        assert "Total images processed: 100" in captured.out

    def test_run_augment_validation_failure(self, capsys):
        """Test augmentation with validation failure."""
        args = MagicMock()
        
        with patch('user_system.main.validate_augment_args', return_value=None):
            run_augment(args)
        
        captured = capsys.readouterr()
        assert "AUGMENTATION MODE ACTIVATED" in captured.out

    def test_run_augment_environment_setup_failure(self, capsys):
        """Test augmentation with environment setup failure."""
        args = MagicMock()
        args.inputs = ["./dataset1"]
        args.workers = 4
        
        mock_data_paths = [Path("./dataset1")]
        mock_config_path = Path("./config.yaml")
        mock_config = {"modes": {"flip_h": {"enabled": True}}}
        
        with patch('user_system.main.validate_augment_args', return_value=(mock_data_paths, mock_config_path, mock_config)), \
             patch('user_system.main.get_base_directory', return_value=Path("./base")), \
             patch('user_system.main.setup_augmentation_environment', return_value=None):
            run_augment(args)
        
        captured = capsys.readouterr()
        assert "Failed to set up augmentation environment" in captured.out


class TestRunLabels:
    """Test cases for run_labels function."""

    def test_run_labels_success_copy_mode(self, capsys):
        """Test successful label modification in copy mode."""
        args = MagicMock()
        args.data = "./dataset"
        args.old_label = "cow"
        args.new_label = "cattle"
        args.out = "./output"
        args.in_place = False
        
        mock_stats = {'files_processed': 10, 'labels_changed': 15, 'classes_updated': 1}
        
        with patch('user_system.main.validate_labels_args', return_value=Path("./dataset")), \
             patch('user_system.main.modify_dataset_labels', return_value=mock_stats):
            run_labels(args)
        
        captured = capsys.readouterr()
        assert "LABEL MODIFICATION MODE ACTIVATED" in captured.out
        assert "Relabel configuration" in captured.out
        assert "Label modification complete" in captured.out

    def test_run_labels_success_in_place_mode(self, capsys):
        """Test successful label modification in in-place mode."""
        args = MagicMock()
        args.data = "./dataset"
        args.old_label = "cow"
        args.new_label = "cattle"
        args.out = None
        args.in_place = True
        
        mock_stats = {'files_processed': 10, 'labels_changed': 15, 'classes_updated': 1}
        
        with patch('user_system.main.validate_labels_args', return_value=Path("./dataset")), \
             patch('user_system.main.modify_dataset_labels', return_value=mock_stats):
            run_labels(args)
        
        captured = capsys.readouterr()
        assert "mode   : in-place" in captured.out

    def test_run_labels_validation_failure(self, capsys):
        """Test label modification with validation failure."""
        args = MagicMock()
        
        with patch('user_system.main.validate_labels_args', return_value=None):
            run_labels(args)
        
        captured = capsys.readouterr()
        assert "LABEL MODIFICATION MODE ACTIVATED" in captured.out

    def test_run_labels_exception_handling(self, capsys):
        """Test label modification with exception handling."""
        args = MagicMock()
        args.data = "./dataset"
        args.old_label = "cow"
        args.new_label = "cattle"
        args.out = "./output"
        args.in_place = False
        
        with patch('user_system.main.validate_labels_args', return_value=Path("./dataset")), \
             patch('user_system.main.modify_dataset_labels', side_effect=Exception("Test error")):
            run_labels(args)
        
        captured = capsys.readouterr()
        assert "LABEL MODIFICATION FAILED" in captured.out
        assert "Error: Test error" in captured.out


class TestRunRecommend:
    """Test cases for run_recommend function."""

    def test_run_recommend_success(self, capsys):
        """Test successful hardware recommendation."""
        args = MagicMock()
        
        with patch('user_system.main.cpu_info', return_value="CPU: Intel Core i7-10700K"), \
             patch('user_system.main.ram_info', return_value="RAM: 16.0 GB"), \
             patch('user_system.main.gpu_info', return_value=["GPU: NVIDIA GeForce RTX 3070 (8 GB VRAM)"]):
            run_recommend(args)
        
        captured = capsys.readouterr()
        assert "HARDWARE CONFIGURATION RECOMMENDATIONS" in captured.out
        assert "YOUR CURRENT SYSTEM" in captured.out
        assert "QUICK PRESET" in captured.out
        assert "BALANCED PRESET" in captured.out
        assert "STRONG PRESET" in captured.out
        assert "RECOMMENDATION FOR YOUR SYSTEM" in captured.out

    def test_run_recommend_hardware_detection_failure(self, capsys):
        """Test hardware recommendation with detection failure."""
        args = MagicMock()
        
        with patch('user_system.main.cpu_info', side_effect=Exception("Detection failed")), \
             patch('user_system.main.ram_info', side_effect=Exception("Detection failed")), \
             patch('user_system.main.gpu_info', side_effect=Exception("Detection failed")):
            run_recommend(args)
        
        captured = capsys.readouterr()
        assert "Hardware detection failed" in captured.out

    def test_run_recommend_auto_recommendation(self, capsys):
        """Test auto-recommendation based on detected hardware."""
        args = MagicMock()
        
        with patch('user_system.main.cpu_info', return_value="CPU: Intel Core i7-10700K"), \
             patch('user_system.main.ram_info', return_value="RAM: 32.0 GB"), \
             patch('user_system.main.gpu_info', return_value=["GPU: NVIDIA GeForce RTX 3070 (VRAM: 8 GB)"]):
            run_recommend(args)
        
        captured = capsys.readouterr()
        assert "STRONG preset recommended" in captured.out

    def test_run_recommend_auto_recommendation_failure(self, capsys):
        """Test auto-recommendation failure."""
        args = MagicMock()
        
        with patch('user_system.main.cpu_info', return_value="CPU: Intel Core i7-10700K"), \
             patch('user_system.main.ram_info', return_value="Invalid format"), \
             patch('user_system.main.gpu_info', return_value=["GPU: NVIDIA GeForce RTX 3070 (8 GB VRAM)"]):
            run_recommend(args)
        
        captured = capsys.readouterr()
        assert "Unable to auto-recommend" in captured.out


class TestMain:
    """Test cases for main function."""

    def test_main_no_args_shows_help(self, capsys):
        """Test main function with no arguments shows help."""
        with patch('sys.argv', ['main.py']):
            main()

        captured = capsys.readouterr()
        assert "usage: main.py" in captured.out

    def test_main_help_mode(self, capsys):
        """Test main function with help mode."""
        with patch('sys.argv', ['main.py', 'help']):
            main()
        
        captured = capsys.readouterr()
        assert "HELP MODE ACTIVATED" in captured.out
        assert "training with existing dataset" in captured.out

    def test_main_recommend_mode(self, capsys):
        """Test main function with recommend mode."""
        with patch('sys.argv', ['main.py', 'recommend']), \
             patch('user_system.main.run_recommend'):
            main()

        captured = capsys.readouterr()
        assert "Mode: recommend" in captured.out

    def test_main_compile_mode(self, capsys):
        """Test main function with compile mode."""
        with patch('sys.argv', ['main.py', 'compile', '--compile-inputs', 'dataset1', '--data', 'output']), \
             patch('user_system.main.run_compile'):
            main()

        captured = capsys.readouterr()
        assert "Mode: compile" in captured.out

    def test_main_train_mode(self, capsys):
        """Test main function with train mode."""
        with patch('sys.argv', ['main.py', 'train', '--data', 'dataset', '--epochs', '50']), \
             patch('user_system.main.run_train'):
            main()

        captured = capsys.readouterr()
        assert "Mode: train" in captured.out

    def test_main_eval_mode(self, capsys):
        """Test main function with eval mode."""
        with patch('sys.argv', ['main.py', 'eval', '--data', 'test_data', '--model', 'model.pt']), \
             patch('user_system.main.run_eval'):
            main()

        captured = capsys.readouterr()
        assert "Mode: eval" in captured.out

    def test_main_labels_mode(self, capsys):
        """Test main function with labels mode."""
        with patch('sys.argv', ['main.py', 'labels', '--data', 'dataset', '--old-label', 'cow', '--new-label', 'cattle']), \
             patch('user_system.main.run_labels'):
            main()

        captured = capsys.readouterr()
        assert "Mode: labels" in captured.out

    def test_main_augment_mode(self, capsys):
        """Test main function with augment mode."""
        with patch('sys.argv', ['main.py', 'augment', '--inputs', 'dataset1', '--workers', '4']), \
             patch('user_system.main.run_augment'):
            main()

        captured = capsys.readouterr()
        assert "Mode: augment" in captured.out

    def test_main_no_mode_shows_error(self, capsys):
        """Test main function with no mode shows error."""
        with patch('sys.argv', ['main.py', '--invalid-flag']), \
             patch('argparse.ArgumentParser.parse_known_args', return_value=(MagicMock(mode=None), [])):
            main()

        captured = capsys.readouterr()
        assert "Please choose a mode" in captured.out

    def test_main_unknown_flags(self, capsys):
        """Test main function with unknown flags."""
        with patch('sys.argv', ['main.py', 'train', '--data', 'dataset', '--unknown-flag']), \
             patch('argparse.ArgumentParser.parse_known_args', return_value=(MagicMock(mode='train'), ['--unknown-flag'])):
            main()
        
        captured = capsys.readouterr()
        assert "Ignoring unrecognized flags" in captured.out
        assert "--unknown-flag" in captured.out

    def test_main_parse_error(self, capsys):
        """Test main function with parse error."""
        with patch('sys.argv', ['main.py', 'invalid', 'mode']), \
             patch('argparse.ArgumentParser.parse_known_args', side_effect=SystemExit):
            main()
        
        captured = capsys.readouterr()
        assert "Failed to parse arguments" in captured.out


class TestMainIntegration:
    """Integration tests for main module."""

    def test_full_workflow_compile_and_train(self, capsys):
        """Test full workflow from compilation to training."""
        # Test compile mode
        with patch('sys.argv', ['main.py', 'compile', '--compile-inputs', 'dataset1', '--data', 'output']), \
             patch('user_system.main.run_compile') as mock_compile:
            main()
        
        mock_compile.assert_called_once()
        
        # Test train mode
        with patch('sys.argv', ['main.py', 'train', '--data', 'dataset', '--epochs', '50']), \
             patch('user_system.main.run_train') as mock_train:
            main()
        
        mock_train.assert_called_once()

    def test_error_handling_workflow(self, capsys):
        """Test error handling workflow."""
        # Test with invalid mode
        with patch('sys.argv', ['main.py', 'invalid']), \
             patch('argparse.ArgumentParser.parse_known_args', return_value=(MagicMock(mode='invalid'), [])):
            main()
        
        captured = capsys.readouterr()
        assert "Mode: invalid" in captured.out

    def test_argument_validation_workflow(self, capsys):
        """Test argument validation workflow."""
        # Test compile with missing arguments
        with patch('sys.argv', ['main.py', 'compile']), \
             patch('user_system.main.run_compile') as mock_compile:
            main()
        
        mock_compile.assert_called_once()
        
        # Test train with missing arguments
        with patch('sys.argv', ['main.py', 'train']), \
             patch('user_system.main.run_train') as mock_train:
            main()
        
        mock_train.assert_called_once()

    def test_mode_routing_workflow(self, capsys):
        """Test mode routing workflow."""
        modes = ['help', 'recommend', 'compile', 'train', 'eval', 'labels', 'augment']
        
        for mode in modes:
            with patch('sys.argv', ['main.py', mode]), \
                 patch(f'user_system.main.run_{mode}' if mode != 'help' else 'builtins.print') as mock_run:
                main()
            
            if mode != 'help':
                mock_run.assert_called_once()
