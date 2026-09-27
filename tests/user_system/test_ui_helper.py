"""
Test suite for user_system.utils.ui_helper module.

This module tests the UI helper functionality including
banner display, argument validation, and configuration printing.
"""

import sys
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import pytest

# Put repo root on sys.path
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Import the module under test
from user_system.utils.ui_helper import (
    show_banner,
    print_mode_header,
    validate_compile_args,
    validate_train_args,
    validate_eval_args,
    validate_labels_args,
    validate_augment_args,
    print_augment_config
)


class TestShowBanner:
    """Test cases for show_banner function."""

    def test_show_banner_output(self, capsys):
        """Test that show_banner produces expected output."""
        show_banner()
        captured = capsys.readouterr()
        
        assert "YOLO Livestock Detection System" in captured.out
        assert "USAGE EXAMPLES" in captured.out
        assert "TRAINING MODES" in captured.out
        assert "DATASET LABELS MODE" in captured.out
        assert "AUGMENTATION MODE" in captured.out
        assert "EVALUATION MODE" in captured.out
        assert "HELP MODE" in captured.out

    def test_show_banner_formatting(self, capsys):
        """Test that show_banner has proper formatting."""
        show_banner()
        captured = capsys.readouterr()
        
        # Check for proper separators
        assert "=" * 60 in captured.out
        # Check for emojis and formatting
        assert "🚀" in captured.out
        assert "📖" in captured.out
        assert "👉" in captured.out


class TestPrintModeHeader:
    """Test cases for print_mode_header function."""

    def test_print_mode_header(self, capsys):
        """Test print_mode_header output."""
        args = MagicMock()
        args.mode = "train"
        
        print_mode_header(args)
        captured = capsys.readouterr()
        
        assert "Mode: train" in captured.out
        assert "Arguments:" in captured.out
        assert "-" * 60 in captured.out


class TestValidateCompileArgs:
    """Test cases for validate_compile_args function."""

    def test_validate_compile_args_success(self):
        """Test successful validation of compile arguments."""
        args = MagicMock()
        args.compile_inputs = ["./dataset1", "./dataset2"]
        args.compile_out = "./output"
        args.data = "./data"
        
        with patch('pathlib.Path.exists', return_value=True):
            result = validate_compile_args(args)
        
        assert result is True

    def test_validate_compile_args_no_inputs(self, capsys):
        """Test validation when no compile inputs provided."""
        args = MagicMock()
        args.compile_inputs = None
        
        result = validate_compile_args(args)
        
        assert result is False
        captured = capsys.readouterr()
        assert "Missing required info for compilation" in captured.out

    def test_validate_compile_args_empty_inputs(self, capsys):
        """Test validation when compile inputs is empty list."""
        args = MagicMock()
        args.compile_inputs = []
        
        result = validate_compile_args(args)
        
        assert result is False
        captured = capsys.readouterr()
        assert "Missing required info for compilation" in captured.out

    def test_validate_compile_args_empty_string_inputs(self, capsys):
        """Test validation when compile inputs contains empty strings."""
        args = MagicMock()
        args.compile_inputs = ["", "   ", ""]
        
        result = validate_compile_args(args)
        
        assert result is False
        captured = capsys.readouterr()
        assert "Missing required info for compilation" in captured.out

    def test_validate_compile_args_missing_paths(self, capsys):
        """Test validation when some input paths don't exist."""
        args = MagicMock()
        args.compile_inputs = ["./existing", "./missing"]
        args.compile_out = "./output"
        
        def mock_exists(path):
            return str(path) == "./existing"
        
        with patch('pathlib.Path.exists', side_effect=mock_exists):
            result = validate_compile_args(args)
        
        assert result is False
        captured = capsys.readouterr()
        assert "Some input paths do not exist" in captured.out
        assert "./missing" in captured.out

    def test_validate_compile_args_no_output(self, capsys):
        """Test validation when no output directory specified."""
        args = MagicMock()
        args.compile_inputs = ["./dataset1"]
        args.compile_out = None
        args.data = None
        
        with patch('pathlib.Path.exists', return_value=True):
            result = validate_compile_args(args)
        
        assert result is False
        captured = capsys.readouterr()
        assert "Missing output directory for compilation" in captured.out

    def test_validate_compile_args_exception_handling(self):
        """Test exception handling in validate_compile_args."""
        args = MagicMock()
        args.compile_inputs = ["./dataset1"]
        
        # Mock getattr to raise exception
        with patch('builtins.getattr', side_effect=Exception("Test error")):
            result = validate_compile_args(args)
        
        assert result is False


class TestValidateTrainArgs:
    """Test cases for validate_train_args function."""

    def test_validate_train_args_with_compile_inputs_success(self):
        """Test validation with valid compile inputs."""
        args = MagicMock()
        args.compile_inputs = ["./dataset1", "./dataset2"]
        args.compile_out = "./output"
        args.data = "./data"
        
        with patch('pathlib.Path.exists', return_value=True):
            result = validate_train_args(args)
        
        assert result is True

    def test_validate_train_args_with_compile_inputs_missing_paths(self, capsys):
        """Test validation with missing compile input paths."""
        args = MagicMock()
        args.compile_inputs = ["./existing", "./missing"]
        
        def mock_exists(path):
            return str(path) == "./existing"
        
        with patch('pathlib.Path.exists', side_effect=mock_exists):
            result = validate_train_args(args)
        
        assert result is False
        captured = capsys.readouterr()
        assert "Some input paths for compilation do not exist" in captured.out

    def test_validate_train_args_with_compile_inputs_no_output(self, capsys):
        """Test validation with compile inputs but no output."""
        args = MagicMock()
        args.compile_inputs = ["./dataset1"]
        args.compile_out = None
        args.data = None
        
        with patch('pathlib.Path.exists', return_value=True):
            result = validate_train_args(args)
        
        assert result is False
        captured = capsys.readouterr()
        assert "When using --compile-inputs, also specify an output" in captured.out

    def test_validate_train_args_empty_compile_inputs(self, capsys):
        """Test validation with empty compile inputs."""
        args = MagicMock()
        args.compile_inputs = []
        
        result = validate_train_args(args)
        
        assert result is False
        captured = capsys.readouterr()
        assert "--compile-inputs was provided but no directories were specified" in captured.out

    def test_validate_train_args_no_compile_inputs_no_data(self, capsys):
        """Test validation without compile inputs and no data."""
        args = MagicMock()
        args.compile_inputs = None
        args.data = None
        
        result = validate_train_args(args)
        
        assert result is False
        captured = capsys.readouterr()
        assert "Please provide --data for training" in captured.out

    def test_validate_train_args_no_compile_inputs_with_data(self):
        """Test validation without compile inputs but with data."""
        args = MagicMock()
        args.compile_inputs = None
        args.data = "./dataset"
        
        result = validate_train_args(args)
        
        assert result is True

    def test_validate_train_args_exception_handling(self, capsys):
        """Test exception handling in validate_train_args."""
        args = MagicMock()
        
        with patch('builtins.getattr', side_effect=Exception("Test error")):
            result = validate_train_args(args)
        
        assert result is False
        captured = capsys.readouterr()
        assert "Please provide --data for training" in captured.out


class TestValidateEvalArgs:
    """Test cases for validate_eval_args function."""

    def test_validate_eval_args_success(self):
        """Test successful validation of eval arguments."""
        args = MagicMock()
        args.data = "./test_data"
        
        with patch('pathlib.Path.exists', return_value=True):
            result = validate_eval_args(args)
        
        assert result == Path("./test_data")

    def test_validate_eval_args_no_data(self, capsys):
        """Test validation when no data provided."""
        args = MagicMock()
        args.data = None
        
        result = validate_eval_args(args)
        
        assert result is None
        captured = capsys.readouterr()
        assert "Missing required argument: --data" in captured.out

    def test_validate_eval_args_nonexistent_data(self, capsys):
        """Test validation when data path doesn't exist."""
        args = MagicMock()
        args.data = "./nonexistent"
        
        with patch('pathlib.Path.exists', return_value=False):
            result = validate_eval_args(args)
        
        assert result is None
        captured = capsys.readouterr()
        assert "Test data not found" in captured.out


class TestValidateLabelsArgs:
    """Test cases for validate_labels_args function."""

    def test_validate_labels_args_success(self):
        """Test successful validation of labels arguments."""
        args = MagicMock()
        args.data = "./dataset"
        args.old_label = "cow"
        args.new_label = "cattle"
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('user_system.utils.ui_helper.validate_dataset_structure', return_value=True):
            result = validate_labels_args(args)
        
        assert result == Path("./dataset")

    def test_validate_labels_args_no_data(self, capsys):
        """Test validation when no data provided."""
        args = MagicMock()
        args.data = None
        
        result = validate_labels_args(args)
        
        assert result is None
        captured = capsys.readouterr()
        assert "Label modification requires --data" in captured.out

    def test_validate_labels_args_no_old_label(self, capsys):
        """Test validation when no old label provided."""
        args = MagicMock()
        args.data = "./dataset"
        args.old_label = None
        
        result = validate_labels_args(args)
        
        assert result is None
        captured = capsys.readouterr()
        assert "Label modification requires --old-label" in captured.out

    def test_validate_labels_args_no_new_label(self, capsys):
        """Test validation when no new label provided."""
        args = MagicMock()
        args.data = "./dataset"
        args.old_label = "cow"
        args.new_label = None
        
        result = validate_labels_args(args)
        
        assert result is None
        captured = capsys.readouterr()
        assert "Label modification requires --new-label" in captured.out

    def test_validate_labels_args_nonexistent_data(self, capsys):
        """Test validation when data path doesn't exist."""
        args = MagicMock()
        args.data = "./nonexistent"
        args.old_label = "cow"
        args.new_label = "cattle"
        
        with patch('pathlib.Path.exists', return_value=False):
            result = validate_labels_args(args)
        
        assert result is None
        captured = capsys.readouterr()
        assert "Dataset path not found" in captured.out

    def test_validate_labels_args_invalid_dataset_structure(self, capsys):
        """Test validation when dataset structure is invalid."""
        args = MagicMock()
        args.data = "./dataset"
        args.old_label = "cow"
        args.new_label = "cattle"
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('user_system.utils.ui_helper.validate_dataset_structure', return_value=False):
            result = validate_labels_args(args)
        
        assert result is None
        captured = capsys.readouterr()
        assert "Invalid dataset structure" in captured.out

    def test_validate_labels_args_validator_unavailable(self, capsys):
        """Test validation when dataset validator is unavailable."""
        args = MagicMock()
        args.data = "./dataset"
        args.old_label = "cow"
        args.new_label = "cattle"
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('user_system.utils.ui_helper.validate_dataset_structure', None):
            result = validate_labels_args(args)
        
        assert result is None
        captured = capsys.readouterr()
        assert "Internal error: dataset validator unavailable" in captured.out


class TestValidateAugmentArgs:
    """Test cases for validate_augment_args function."""

    def test_validate_augment_args_success(self):
        """Test successful validation of augment arguments."""
        args = MagicMock()
        args.inputs = ["./dataset1", "./dataset2"]
        args.config = "./config.yaml"
        
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({"modes": {"flip_h": {"enabled": True}}}, f)
            config_path = f.name
        
        try:
            with patch('pathlib.Path.exists', return_value=True), \
                 patch('user_system.utils.ui_helper.validate_dataset_structure', return_value=True), \
                 patch('pathlib.Path.open', mock_open(read_data=yaml.dump({"modes": {"flip_h": {"enabled": True}}}))):
                result = validate_augment_args(args)
            
            assert result is not None
            data_paths, config_path_result, cfg = result
            assert len(data_paths) == 2
            assert "modes" in cfg
        finally:
            Path(config_path).unlink(missing_ok=True)

    def test_validate_augment_args_no_inputs(self, capsys):
        """Test validation when no inputs provided."""
        args = MagicMock()
        args.inputs = None
        
        result = validate_augment_args(args)
        
        assert result is None
        captured = capsys.readouterr()
        assert "Augmentation requires at least one dataset via --inputs" in captured.out

    def test_validate_augment_args_empty_inputs(self, capsys):
        """Test validation when inputs is empty list."""
        args = MagicMock()
        args.inputs = []
        
        result = validate_augment_args(args)
        
        assert result is None
        captured = capsys.readouterr()
        assert "Augmentation requires at least one dataset via --inputs" in captured.out

    def test_validate_augment_args_missing_paths(self, capsys):
        """Test validation when some input paths don't exist."""
        args = MagicMock()
        args.inputs = ["./existing", "./missing"]
        
        def mock_exists(self):
            return str(self) == "./existing"
        
        with patch.object(Path, 'exists', mock_exists):
            result = validate_augment_args(args)
        
        assert result is None
        captured = capsys.readouterr()
        assert "Some inputs do not exist" in captured.out

    def test_validate_augment_args_invalid_datasets(self, capsys):
        """Test validation when some inputs are invalid datasets."""
        args = MagicMock()
        args.inputs = ["./dataset1", "./dataset2"]
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('user_system.utils.ui_helper.validate_dataset_structure', return_value=False):
            result = validate_augment_args(args)
        
        assert result is None
        captured = capsys.readouterr()
        assert "Some inputs are not valid YOLO datasets" in captured.out

    def test_validate_augment_args_config_not_found(self, capsys):
        """Test validation when config file is not found."""
        args = MagicMock()
        args.inputs = ["./dataset1"]
        args.config = "./nonexistent.yaml"
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('user_system.utils.ui_helper.validate_dataset_structure', return_value=True), \
             patch('pathlib.Path.open', side_effect=FileNotFoundError("Config not found")):
            result = validate_augment_args(args)
        
        assert result is None
        captured = capsys.readouterr()
        assert "Failed to load augmentation config" in captured.out

    def test_validate_augment_args_invalid_yaml(self, capsys):
        """Test validation when config file has invalid YAML."""
        args = MagicMock()
        args.inputs = ["./dataset1"]
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('user_system.utils.ui_helper.validate_dataset_structure', return_value=True), \
             patch('pathlib.Path.open', mock_open(read_data="invalid: yaml: content: [")):
            result = validate_augment_args(args)
        
        assert result is None
        captured = capsys.readouterr()
        assert "Failed to load augmentation config" in captured.out

    def test_validate_augment_args_missing_modes_key(self, capsys):
        """Test validation when config is missing 'modes' key."""
        args = MagicMock()
        args.inputs = ["./dataset1"]
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('user_system.utils.ui_helper.validate_dataset_structure', return_value=True), \
             patch('pathlib.Path.open', mock_open(read_data=yaml.dump({"output": "test"}))):
            result = validate_augment_args(args)
        
        assert result is None
        captured = capsys.readouterr()
        assert "Invalid config: missing 'modes' key" in captured.out


class TestPrintAugmentConfig:
    """Test cases for print_augment_config function."""

    def test_print_augment_config_basic(self, capsys):
        """Test printing basic augmentation config."""
        cfg = {
            "output": "augmented",
            "in_place": False,
            "seed": 42,
            "modes": {
                "flip_h": {"enabled": True},
                "rotate": {"enabled": False}
            }
        }
        
        print_augment_config(cfg)
        captured = capsys.readouterr()
        
        assert "AUGMENTATION CONFIGURATION" in captured.out
        assert "General Settings:" in captured.out
        assert "Augmentation Modes:" in captured.out
        assert "ENABLED:" in captured.out
        assert "DISABLED:" in captured.out
        assert "flip_h" in captured.out
        assert "rotate" in captured.out

    def test_print_augment_config_with_parameters(self, capsys):
        """Test printing config with mode parameters."""
        cfg = {
            "modes": {
                "rotate": {
                    "enabled": True,
                    "prob": 0.5,
                    "max_deg": 15
                },
                "translate": {
                    "enabled": True,
                    "max_frac": 0.1
                },
                "hsv": {
                    "enabled": True,
                    "h": 0.015,
                    "s": 0.7,
                    "v": 0.4
                }
            }
        }
        
        print_augment_config(cfg)
        captured = capsys.readouterr()
        
        assert "max_deg: 15" in captured.out
        assert "max_frac: 0.1" in captured.out
        assert "HSV(h: 0.015, s: 0.7, v: 0.4)" in captured.out

    def test_print_augment_config_simple_boolean_modes(self, capsys):
        """Test printing config with simple boolean mode configs."""
        cfg = {
            "modes": {
                "flip_h": True,
                "grayscale": False
            }
        }
        
        print_augment_config(cfg)
        captured = capsys.readouterr()
        
        assert "flip_h" in captured.out
        assert "grayscale" in captured.out

    def test_print_augment_config_empty_modes(self, capsys):
        """Test printing config with empty modes."""
        cfg = {
            "output": "test",
            "modes": {}
        }
        
        print_augment_config(cfg)
        captured = capsys.readouterr()
        
        assert "AUGMENTATION CONFIGURATION" in captured.out
        assert "General Settings:" in captured.out
        assert "output" in captured.out

    def test_print_augment_config_no_modes(self, capsys):
        """Test printing config without modes key."""
        cfg = {
            "output": "test",
            "seed": 42
        }
        
        print_augment_config(cfg)
        captured = capsys.readouterr()
        
        assert "AUGMENTATION CONFIGURATION" in captured.out
        assert "General Settings:" in captured.out
        assert "output" in captured.out
        assert "seed" in captured.out


class TestUIHelperIntegration:
    """Integration tests for UI helper functions."""

    def test_full_validation_workflow(self):
        """Test complete validation workflow."""
        # Test compile validation
        args = MagicMock()
        args.compile_inputs = ["./dataset1", "./dataset2"]
        args.compile_out = "./output"
        
        with patch('pathlib.Path.exists', return_value=True):
            compile_result = validate_compile_args(args)
        
        assert compile_result is True
        
        # Test train validation
        args.compile_inputs = ["./dataset1"]
        args.data = "./data"
        
        with patch('pathlib.Path.exists', return_value=True):
            train_result = validate_train_args(args)
        
        assert train_result is True
        
        # Test eval validation
        args.data = "./test_data"
        
        with patch('pathlib.Path.exists', return_value=True):
            eval_result = validate_eval_args(args)
        
        assert eval_result == Path("./test_data")

    def test_banner_and_header_integration(self, capsys):
        """Test banner and header functions work together."""
        show_banner()
        print()  # Add newline
        
        args = MagicMock()
        args.mode = "train"
        print_mode_header(args)
        
        captured = capsys.readouterr()
        
        assert "YOLO Livestock Detection System" in captured.out
        assert "Mode: train" in captured.out
        assert "Arguments:" in captured.out
