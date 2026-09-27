"""
Test suite for user_system.utils.arg_helper module.

This module tests the argument helper utilities for dataset preparation,
compilation, and management functionality.
"""

import sys
import json
import tempfile
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import pytest
import yaml

# Put repo root on sys.path
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

def _ensure_utils_shim():
    """Ensure utils modules are available for testing."""
    import types
    if "utils" not in sys.modules:
        pkg = types.ModuleType("utils")
        pkg.__path__ = []
        sys.modules["utils"] = pkg
    pkg = sys.modules["utils"]
    for sub in ("compile_helper", "dataset_shuffle"):
        try:
            m = __import__(f"user_system.utils.{sub}", fromlist=["*"])
            sys.modules[f"utils.{sub}"] = m
            setattr(pkg, sub, m)
        except Exception:
            pass

_ensure_utils_shim()

# Import the module under test
from user_system.utils.arg_helper import (
    find_dataset_yaml,
    parse_class_names,
    ensure_all_exist,
    run_compile_subprocess,
    reshuffle_or_compile,
    prepare_dataset
)


class TestFindDatasetYaml:
    """Test cases for find_dataset_yaml function."""

    def test_find_dataset_yaml_data_yaml(self):
        """Test finding data.yaml file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            yaml_file = temp_path / "data.yaml"
            yaml_file.write_text("test: data")
            
            result = find_dataset_yaml(temp_path)
            assert result == yaml_file

    def test_find_dataset_yaml_dataset_yaml(self):
        """Test finding dataset.yaml file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            yaml_file = temp_path / "dataset.yaml"
            yaml_file.write_text("test: data")
            
            result = find_dataset_yaml(temp_path)
            assert result == yaml_file

    def test_find_dataset_yaml_dataset_yml(self):
        """Test finding dataset.yml file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            yaml_file = temp_path / "dataset.yml"
            yaml_file.write_text("test: data")
            
            result = find_dataset_yaml(temp_path)
            assert result == yaml_file

    def test_find_dataset_yaml_data_yml(self):
        """Test finding data.yml file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            yaml_file = temp_path / "data.yml"
            yaml_file.write_text("test: data")
            
            result = find_dataset_yaml(temp_path)
            assert result == yaml_file

    def test_find_dataset_yaml_wildcard_search(self):
        """Test wildcard search for any yaml/yml file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            yaml_file = temp_path / "custom.yaml"
            yaml_file.write_text("test: data")
            
            result = find_dataset_yaml(temp_path)
            assert result == yaml_file

    def test_find_dataset_yaml_no_files(self):
        """Test when no yaml files exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            result = find_dataset_yaml(temp_path)
            assert result is None

    def test_find_dataset_yaml_priority_order(self):
        """Test that common names are found before wildcard search."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            # Create both data.yaml and custom.yaml
            data_yaml = temp_path / "data.yaml"
            custom_yaml = temp_path / "custom.yaml"
            data_yaml.write_text("test: data")
            custom_yaml.write_text("test: custom")
            
            result = find_dataset_yaml(temp_path)
            assert result == data_yaml  # Should prefer data.yaml


class TestParseClassNames:
    """Test cases for parse_class_names function."""

    def test_parse_class_names_none(self):
        """Test parsing when args_names is None."""
        result = parse_class_names(None)
        assert result is None

    def test_parse_class_names_empty_string(self):
        """Test parsing when args_names is empty string."""
        result = parse_class_names("")
        assert result is None

    def test_parse_class_names_success(self):
        """Test successful parsing of class names."""
        with patch('user_system.utils.arg_helper.parse_names_arg') as mock_parse:
            mock_parse.return_value = {"class1": "cow", "class2": "chicken"}
            
            result = parse_class_names('{"class1": "cow"}')
            assert result == {"class1": "cow", "class2": "chicken"}
            mock_parse.assert_called_once_with('{"class1": "cow"}')

    def test_parse_class_names_exception(self):
        """Test parsing when an exception occurs."""
        with patch('user_system.utils.arg_helper.parse_names_arg') as mock_parse:
            mock_parse.side_effect = ValueError("Invalid JSON")
            
            result = parse_class_names('invalid json')
            assert result is None
            mock_parse.assert_called_once_with('invalid json')


class TestEnsureAllExist:
    """Test cases for ensure_all_exist function."""

    def test_ensure_all_exist_success(self):
        """Test when all paths exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            path1 = temp_path / "dataset1"
            path2 = temp_path / "dataset2"
            path1.mkdir()
            path2.mkdir()
            
            # Should not raise any exception
            ensure_all_exist([path1, path2])

    def test_ensure_all_exist_missing_path(self):
        """Test when a path doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            path1 = temp_path / "dataset1"
            path2 = temp_path / "nonexistent"
            path1.mkdir()
            
            with pytest.raises(SystemExit) as exc_info:
                ensure_all_exist([path1, path2])
            assert "Input dataset 2 not found" in str(exc_info.value)

    def test_ensure_all_exist_empty_list(self):
        """Test with empty list."""
        # Should not raise any exception
        ensure_all_exist([])


class TestRunCompileSubprocess:
    """Test cases for run_compile_subprocess function."""

    def test_run_compile_subprocess_success(self):
        """Test successful subprocess execution."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            out_dir = temp_path / "output"
            inputs = [temp_path / "input1", temp_path / "input2"]
            
            # Create input directories
            for inp in inputs:
                inp.mkdir()
            
            with patch('subprocess.run') as mock_run:
                mock_proc = MagicMock()
                mock_proc.returncode = 0
                mock_proc.stdout = "Compilation successful\n{\"stats\": \"data\"}"
                mock_run.return_value = mock_proc
                
                result_dir, stats = run_compile_subprocess(inputs, out_dir)
                
                assert result_dir == out_dir
                assert stats == {"stats": "data"}
                mock_run.assert_called_once()

    def test_run_compile_subprocess_with_names(self):
        """Test subprocess execution with names parameter."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            out_dir = temp_path / "output"
            inputs = [temp_path / "input1"]
            inputs[0].mkdir()
            names = {"class1": "cow"}
            
            with patch('subprocess.run') as mock_run:
                mock_proc = MagicMock()
                mock_proc.returncode = 0
                mock_proc.stdout = "Compilation successful"
                mock_run.return_value = mock_proc
                
                result_dir, stats = run_compile_subprocess(inputs, out_dir, names)
                
                assert result_dir == out_dir
                mock_run.assert_called_once()

    def test_run_compile_subprocess_failure(self):
        """Test subprocess execution failure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            out_dir = temp_path / "output"
            inputs = [temp_path / "input1"]
            inputs[0].mkdir()
            
            with patch('subprocess.run') as mock_run:
                mock_proc = MagicMock()
                mock_proc.returncode = 1
                mock_proc.stdout = "Error output"
                mock_proc.stderr = "Error message"
                mock_run.return_value = mock_proc
                
                with pytest.raises(RuntimeError) as exc_info:
                    run_compile_subprocess(inputs, out_dir)
                assert "compile.py failed" in str(exc_info.value)

    def test_run_compile_subprocess_with_flag_helper(self):
        """Test subprocess execution with flag_helper.yaml configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            out_dir = temp_path / "output"
            inputs = [temp_path / "input1"]
            inputs[0].mkdir()
            
            # Create flag_helper.yaml
            flag_file = temp_path / "configs" / "flag_helper.yaml"
            flag_file.parent.mkdir()
            flag_file.write_text("""
keep_classes: [0, 1, 2]
aliases:
  cow: [cattle, COW]
""")
            
            with patch('subprocess.run') as mock_run, \
                 patch('user_system.utils.arg_helper.Path') as mock_path:
                
                # Mock the flag file path resolution
                mock_path.side_effect = lambda x: flag_file if "flag_helper.yaml" in str(x) else Path(x)
                
                mock_proc = MagicMock()
                mock_proc.returncode = 0
                mock_proc.stdout = "Compilation successful"
                mock_run.return_value = mock_proc
                
                result_dir, stats = run_compile_subprocess(inputs, out_dir)
                
                assert result_dir == out_dir
                mock_run.assert_called_once()

    def test_run_compile_subprocess_no_stats(self):
        """Test subprocess execution without stats in output."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            out_dir = temp_path / "output"
            inputs = [temp_path / "input1"]
            inputs[0].mkdir()
            
            with patch('subprocess.run') as mock_run:
                mock_proc = MagicMock()
                mock_proc.returncode = 0
                mock_proc.stdout = "Compilation successful (no JSON stats)"
                mock_run.return_value = mock_proc
                
                result_dir, stats = run_compile_subprocess(inputs, out_dir)
                
                assert result_dir == out_dir
                assert stats is None


class TestReshuffleOrCompile:
    """Test cases for reshuffle_or_compile function."""

    def test_reshuffle_or_compile_dataset_shuffle_true(self):
        """Test when dataset_shuffle is True."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            out_dir = temp_path / "output"
            inputs = [temp_path / "input1"]
            inputs[0].mkdir()
            
            with patch('user_system.utils.arg_helper.reshuffle_datasets') as mock_reshuffle:
                mock_reshuffle.return_value = {"dataset1": {"train": 10, "val": 2, "test": 1}}
                
                result_dir, stats = reshuffle_or_compile(
                    inputs=inputs,
                    out_dir=out_dir,
                    names=None,
                    dataset_shuffle=True,
                    shuffle_seed=42
                )
                
                expected_dir = out_dir.resolve() / "unified_from_shuffled"
                assert result_dir.resolve() == expected_dir
                assert stats == {"reshuffle_stats": {"dataset1": {"train": 10, "val": 2, "test": 1}}}
                mock_reshuffle.assert_called_once_with(inputs, out_dir.resolve(), seed=42)

    def test_reshuffle_or_compile_dataset_shuffle_false(self):
        """Test when dataset_shuffle is False."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            out_dir = temp_path / "output"
            inputs = [temp_path / "input1"]
            inputs[0].mkdir()
            
            with patch('user_system.utils.arg_helper.run_compile_subprocess') as mock_compile:
                mock_compile.return_value = (out_dir, {"stats": "data"})
                
                result_dir, stats = reshuffle_or_compile(
                    inputs=inputs,
                    out_dir=out_dir,
                    names={"class1": "cow"},
                    dataset_shuffle=False
                )
                
                assert result_dir.resolve() == out_dir.resolve()
                assert stats == {"stats": "data"}
                mock_compile.assert_called_once_with(inputs, out_dir.resolve(), {"class1": "cow"})

    def test_reshuffle_or_compile_with_fallback_data(self):
        """Test with fallback_data parameter."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            inputs = [temp_path / "input1"]
            inputs[0].mkdir()
            
            with patch('user_system.utils.arg_helper.run_compile_subprocess') as mock_compile:
                mock_compile.return_value = (Path("./unified"), {"stats": "data"})
                
                result_dir, stats = reshuffle_or_compile(
                    inputs=inputs,
                    out_dir=Path("./unified"),  # Use fallback_data as out_dir
                    names=None,
                    dataset_shuffle=False,
                    fallback_data="./unified"
                )
                
                assert result_dir == Path("./unified")
                assert stats == {"stats": "data"}


class TestPrepareDataset:
    """Test cases for prepare_dataset function."""

    def test_prepare_dataset_with_compile_inputs(self):
        """Test dataset preparation with compile_inputs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create mock args object
            args = MagicMock()
            args.compile_inputs = [str(temp_path / "input1"), str(temp_path / "input2")]
            args.compile_out = str(temp_path / "output")
            args.names = '{"class1": "cow"}'
            args.dataset_shuffle = False
            args.shuffle_seed = None
            
            # Create input directories
            for inp in args.compile_inputs:
                Path(inp).mkdir()
            
            with patch('user_system.utils.arg_helper.parse_class_names') as mock_parse, \
                 patch('user_system.utils.arg_helper.ensure_all_exist') as mock_ensure, \
                 patch('user_system.utils.arg_helper.reshuffle_or_compile') as mock_reshuffle:
                
                mock_parse.return_value = {"class1": "cow"}
                mock_reshuffle.return_value = (temp_path / "output", {"stats": "data"})
                
                result = prepare_dataset(args)
                
                assert result == temp_path / "output"
                mock_parse.assert_called_once_with('{"class1": "cow"}')
                mock_ensure.assert_called_once()
                mock_reshuffle.assert_called_once()

    def test_prepare_dataset_with_dataset_shuffle(self):
        """Test dataset preparation with dataset_shuffle enabled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create mock args object
            args = MagicMock()
            args.compile_inputs = [str(temp_path / "input1")]
            args.compile_out = str(temp_path / "output")
            args.names = None
            args.dataset_shuffle = True
            args.shuffle_seed = 42
            
            # Create input directory
            Path(args.compile_inputs[0]).mkdir()
            
            with patch('user_system.utils.arg_helper.parse_class_names') as mock_parse, \
                 patch('user_system.utils.arg_helper.ensure_all_exist') as mock_ensure, \
                 patch('user_system.utils.arg_helper.reshuffle_or_compile') as mock_reshuffle:
                
                mock_parse.return_value = None
                mock_reshuffle.return_value = (temp_path / "output" / "unified_from_shuffled", {"reshuffle_stats": {}})
                
                result = prepare_dataset(args)
                
                assert result == temp_path / "output" / "unified_from_shuffled"
                mock_reshuffle.assert_called_once()

    def test_prepare_dataset_existing_dataset(self):
        """Test using existing dataset without compilation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            existing_dataset = temp_path / "existing_dataset"
            existing_dataset.mkdir()
            
            # Create mock args object
            args = MagicMock()
            args.compile_inputs = None
            args.data = str(existing_dataset)
            args.dataset_shuffle = False
            
            # Remove compile_out attribute to avoid confusion
            del args.compile_out
            
            result = prepare_dataset(args)
            
            assert result == existing_dataset

    def test_prepare_dataset_existing_dataset_with_shuffle(self):
        """Test reshuffling existing dataset."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            existing_dataset = temp_path / "existing_dataset"
            existing_dataset.mkdir()
            
            # Create mock args object
            args = MagicMock()
            args.compile_inputs = None
            args.data = str(existing_dataset)
            args.dataset_shuffle = True
            args.shuffle_seed = 42
            
            # Set compile_out to the same as data to avoid path confusion
            args.compile_out = str(existing_dataset)
            
            with patch('user_system.utils.arg_helper.reshuffle_datasets') as mock_reshuffle:
                mock_reshuffle.return_value = {"existing_dataset": {"train": 10, "val": 2, "test": 1}}
                
                result = prepare_dataset(args)
                
                expected_result = existing_dataset.resolve() / "unified_from_shuffled"
                assert result.resolve() == expected_result
                mock_reshuffle.assert_called_once()

    def test_prepare_dataset_missing_dataset(self):
        """Test when dataset doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            missing_dataset = temp_path / "missing_dataset"
            
            # Create mock args object
            args = MagicMock()
            args.compile_inputs = None
            args.data = str(missing_dataset)
            args.dataset_shuffle = False
            
            with pytest.raises(SystemExit) as exc_info:
                prepare_dataset(args)
            assert "Dataset not found" in str(exc_info.value)

    def test_prepare_dataset_no_compile_out(self):
        """Test when compile_out is not available."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            existing_dataset = temp_path / "existing_dataset"
            existing_dataset.mkdir()
            
            # Create mock args object
            args = MagicMock()
            args.compile_inputs = None
            args.data = str(existing_dataset)
            args.dataset_shuffle = False
            
            # Remove compile_out attribute
            del args.compile_out
            
            result = prepare_dataset(args)
            
            assert result == existing_dataset

    def test_prepare_dataset_with_stats_printing(self):
        """Test dataset preparation with stats printing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create mock args object
            args = MagicMock()
            args.compile_inputs = [str(temp_path / "input1")]
            args.compile_out = str(temp_path / "output")
            args.names = None
            args.dataset_shuffle = True
            args.shuffle_seed = None
            
            # Create input directory
            Path(args.compile_inputs[0]).mkdir()
            
            with patch('user_system.utils.arg_helper.parse_class_names') as mock_parse, \
                 patch('user_system.utils.arg_helper.ensure_all_exist') as mock_ensure, \
                 patch('user_system.utils.arg_helper.reshuffle_or_compile') as mock_reshuffle:
                
                mock_parse.return_value = None
                mock_reshuffle.return_value = (
                    temp_path / "output" / "unified_from_shuffled",
                    {
                        "reshuffle_stats": {
                            "dataset1": {"train": 10, "val": 2, "test": 1},
                            "dataset2": {"train": 20, "val": 4, "test": 2},
                            "_compiled_out": {"path": str(temp_path / "output" / "unified_from_shuffled")}
                        }
                    }
                )
                
                result = prepare_dataset(args)
                
                assert result == temp_path / "output" / "unified_from_shuffled"
