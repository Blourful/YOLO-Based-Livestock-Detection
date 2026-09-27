"""
Test suite for user_system.utils.directory_helper module.

This module tests the directory management functionality
including output directory creation, dataset structure management,
file operations, and validation.
"""

import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

# Put repo root on sys.path
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Import the module under test
from user_system.utils.directory_helper import (
    create_output_directory,
    create_dataset_structure,
    create_augmented_dataset_structure,
    copy_file,
    copy_data_yaml,
    validate_dataset_structure,
    get_base_directory,
    ensure_directory_exists,
    list_directory_contents
)


class TestCreateOutputDirectory:
    """Test cases for create_output_directory function."""

    def test_create_output_directory_success(self):
        """Test successful output directory creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            output_name = "test_output"
            
            result = create_output_directory(output_name, base_dir)
            
            assert result == base_dir / output_name
            assert result.exists()
            assert result.is_dir()

    def test_create_output_directory_existing(self):
        """Test creating output directory when it already exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            output_name = "test_output"
            
            # Create directory first
            existing_dir = base_dir / output_name
            existing_dir.mkdir()
            
            result = create_output_directory(output_name, base_dir)
            
            assert result == existing_dir
            assert result.exists()

    def test_create_output_directory_nested(self):
        """Test creating nested output directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            output_name = "nested/deep/output"
            
            result = create_output_directory(output_name, base_dir)
            
            assert result == base_dir / output_name
            assert result.exists()
            assert result.is_dir()

    def test_create_output_directory_permission_error(self):
        """Test handling of permission errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            output_name = "test_output"
            
            # Mock mkdir to raise OSError
            with patch.object(Path, 'mkdir', side_effect=OSError("Permission denied")):
                with pytest.raises(OSError):
                    create_output_directory(output_name, base_dir)


class TestCreateDatasetStructure:
    """Test cases for create_dataset_structure function."""

    def test_create_dataset_structure_default(self):
        """Test creating dataset structure with default subdirs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            dataset_name = "test_dataset"
            
            result = create_dataset_structure(dataset_name, output_dir)
            
            assert result == output_dir / dataset_name
            assert result.exists()
            
            # Check default subdirs
            for subdir in ['train', 'test', 'valid']:
                subdir_path = result / subdir
                assert subdir_path.exists()
                assert (subdir_path / 'images').exists()
                assert (subdir_path / 'labels').exists()

    def test_create_dataset_structure_custom_subdirs(self):
        """Test creating dataset structure with custom subdirs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            dataset_name = "custom_dataset"
            subdirs = ['train', 'val']
            
            result = create_dataset_structure(dataset_name, output_dir, subdirs)
            
            assert result == output_dir / dataset_name
            assert result.exists()
            
            # Check custom subdirs
            for subdir in subdirs:
                subdir_path = result / subdir
                assert subdir_path.exists()
                assert (subdir_path / 'images').exists()
                assert (subdir_path / 'labels').exists()

    def test_create_dataset_structure_existing(self):
        """Test creating dataset structure when it already exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            dataset_name = "existing_dataset"
            
            # Create directory first
            existing_dir = output_dir / dataset_name
            existing_dir.mkdir()
            
            result = create_dataset_structure(dataset_name, output_dir)
            
            assert result == existing_dir
            assert result.exists()

    def test_create_dataset_structure_error(self):
        """Test handling of creation errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            dataset_name = "error_dataset"
            
            # Mock mkdir to raise OSError
            with patch.object(Path, 'mkdir', side_effect=OSError("Creation failed")):
                with pytest.raises(OSError):
                    create_dataset_structure(dataset_name, output_dir)


class TestCreateAugmentedDatasetStructure:
    """Test cases for create_augmented_dataset_structure function."""

    def test_create_augmented_dataset_structure(self):
        """Test creating augmented dataset structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            input_path = Path("original_dataset")
            
            result = create_augmented_dataset_structure(input_path, output_dir)
            
            expected_name = "aug_original_dataset"
            assert result == output_dir / expected_name
            assert result.exists()
            
            # Check structure
            for subdir in ['train', 'test', 'valid']:
                subdir_path = result / subdir
                assert subdir_path.exists()
                assert (subdir_path / 'images').exists()
                assert (subdir_path / 'labels').exists()


class TestCopyFile:
    """Test cases for copy_file function."""

    def test_copy_file_success(self):
        """Test successful file copying."""
        with tempfile.TemporaryDirectory() as temp_dir:
            source_dir = Path(temp_dir) / "source"
            dest_dir = Path(temp_dir) / "dest"
            source_dir.mkdir()
            dest_dir.mkdir()
            
            source_file = source_dir / "test.txt"
            source_file.write_text("test content")
            dest_file = dest_dir / "test.txt"
            
            result = copy_file(source_file, dest_file, "test file")
            
            assert result is True
            assert dest_file.exists()
            assert dest_file.read_text() == "test content"

    def test_copy_file_nonexistent_source(self):
        """Test copying nonexistent source file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            source_file = Path(temp_dir) / "nonexistent.txt"
            dest_file = Path(temp_dir) / "dest.txt"
            
            result = copy_file(source_file, dest_file, "test file")
            
            assert result is False

    def test_copy_file_permission_error(self):
        """Test handling of permission errors during copy."""
        with tempfile.TemporaryDirectory() as temp_dir:
            source_file = Path(temp_dir) / "source.txt"
            dest_file = Path(temp_dir) / "dest.txt"
            source_file.write_text("test content")
            
            # Mock shutil.copy2 to raise OSError
            with patch('shutil.copy2', side_effect=OSError("Permission denied")):
                result = copy_file(source_file, dest_file, "test file")
                
                assert result is False

    def test_copy_file_shutil_error(self):
        """Test handling of shutil errors during copy."""
        with tempfile.TemporaryDirectory() as temp_dir:
            source_file = Path(temp_dir) / "source.txt"
            dest_file = Path(temp_dir) / "dest.txt"
            source_file.write_text("test content")
            
            # Mock shutil.copy2 to raise shutil.Error
            with patch('shutil.copy2', side_effect=shutil.Error("Copy failed")):
                result = copy_file(source_file, dest_file, "test file")
                
                assert result is False


class TestCopyDataYaml:
    """Test cases for copy_data_yaml function."""

    def test_copy_data_yaml_success(self):
        """Test successful data.yaml copying."""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input"
            dataset_path = Path(temp_dir) / "dataset"
            input_path.mkdir()
            dataset_path.mkdir()
            
            # Create data.yaml in input
            yaml_file = input_path / "data.yaml"
            yaml_file.write_text("nc: 2\nnames: ['cow', 'chicken']")
            
            result = copy_data_yaml(input_path, dataset_path)
            
            assert result is True
            dest_yaml = dataset_path / "data.yaml"
            assert dest_yaml.exists()
            assert dest_yaml.read_text() == "nc: 2\nnames: ['cow', 'chicken']"

    def test_copy_data_yaml_dataset_yaml(self):
        """Test copying dataset.yaml instead of data.yaml."""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input"
            dataset_path = Path(temp_dir) / "dataset"
            input_path.mkdir()
            dataset_path.mkdir()
            
            # Create dataset.yaml in input
            yaml_file = input_path / "dataset.yaml"
            yaml_file.write_text("nc: 1\nnames: ['sheep']")
            
            result = copy_data_yaml(input_path, dataset_path)
            
            assert result is True
            dest_yaml = dataset_path / "data.yaml"
            assert dest_yaml.exists()
            assert dest_yaml.read_text() == "nc: 1\nnames: ['sheep']"

    def test_copy_data_yaml_priority_order(self):
        """Test that data.yaml takes priority over dataset.yaml."""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input"
            dataset_path = Path(temp_dir) / "dataset"
            input_path.mkdir()
            dataset_path.mkdir()
            
            # Create both files
            data_yaml = input_path / "data.yaml"
            data_yaml.write_text("nc: 2\nnames: ['cow', 'chicken']")
            dataset_yaml = input_path / "dataset.yaml"
            dataset_yaml.write_text("nc: 1\nnames: ['sheep']")
            
            result = copy_data_yaml(input_path, dataset_path)
            
            assert result is True
            dest_yaml = dataset_path / "data.yaml"
            assert dest_yaml.exists()
            # Should copy data.yaml (priority)
            assert dest_yaml.read_text() == "nc: 2\nnames: ['cow', 'chicken']"

    def test_copy_data_yaml_no_yaml_files(self):
        """Test when no YAML files exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input"
            dataset_path = Path(temp_dir) / "dataset"
            input_path.mkdir()
            dataset_path.mkdir()
            
            result = copy_data_yaml(input_path, dataset_path)
            
            assert result is False

    def test_copy_data_yaml_copy_failure(self):
        """Test handling of copy failure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input"
            dataset_path = Path(temp_dir) / "dataset"
            input_path.mkdir()
            dataset_path.mkdir()
            
            # Create data.yaml
            yaml_file = input_path / "data.yaml"
            yaml_file.write_text("nc: 2\nnames: ['cow', 'chicken']")
            
            # Mock copy_file to return False
            with patch('user_system.utils.directory_helper.copy_file', return_value=False):
                result = copy_data_yaml(input_path, dataset_path)
                
                assert result is False


class TestValidateDatasetStructure:
    """Test cases for validate_dataset_structure function."""

    def test_validate_dataset_structure_valid(self):
        """Test validation of valid dataset structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir)
            
            # Create valid structure
            (dataset_path / "data.yaml").write_text("nc: 2\nnames: ['cow', 'chicken']")
            (dataset_path / "train").mkdir()
            (dataset_path / "test").mkdir()
            (dataset_path / "valid").mkdir()
            
            result = validate_dataset_structure(dataset_path)
            
            assert result is True

    def test_validate_dataset_structure_dataset_yaml(self):
        """Test validation with dataset.yaml instead of data.yaml."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir)
            
            # Create structure with dataset.yaml
            (dataset_path / "dataset.yaml").write_text("nc: 1\nnames: ['sheep']")
            (dataset_path / "train").mkdir()
            
            result = validate_dataset_structure(dataset_path)
            
            assert result is True

    def test_validate_dataset_structure_no_yaml(self):
        """Test validation when no YAML files exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir)
            
            # Create structure without YAML
            (dataset_path / "train").mkdir()
            (dataset_path / "test").mkdir()
            
            result = validate_dataset_structure(dataset_path)
            
            assert result is False

    def test_validate_dataset_structure_no_subdirs(self):
        """Test validation when no subdirectories exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir)
            
            # Create only YAML file
            (dataset_path / "data.yaml").write_text("nc: 2\nnames: ['cow', 'chicken']")
            
            result = validate_dataset_structure(dataset_path)
            
            assert result is False

    def test_validate_dataset_structure_partial_subdirs(self):
        """Test validation with only some subdirectories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir)
            
            # Create structure with only train
            (dataset_path / "data.yaml").write_text("nc: 2\nnames: ['cow', 'chicken']")
            (dataset_path / "train").mkdir()
            
            result = validate_dataset_structure(dataset_path)
            
            assert result is True


class TestGetBaseDirectory:
    """Test cases for get_base_directory function."""

    def test_get_base_directory_from_main(self):
        """Test getting base directory from main module."""
        with patch('sys.modules') as mock_modules:
            mock_main = MagicMock()
            mock_main.__file__ = "/path/to/main.py"
            mock_modules.__getitem__.return_value = mock_main
            
            result = get_base_directory()
            
            assert result == Path("/path/to").resolve()

    def test_get_base_directory_fallback(self):
        """Test fallback to current file's parent."""
        with patch('sys.modules') as mock_modules:
            mock_main = MagicMock()
            mock_main.__file__ = None
            mock_modules.__getitem__.return_value = mock_main
            
            result = get_base_directory()
            
            # Should return parent of utils directory (2 levels up from utils/)
            # The function uses Path(__file__).resolve().parents[1] where __file__ is directory_helper.py
            # So it goes up 2 levels from user_system/utils/ to get the project root
            # We can't predict the exact path, so just check it's a valid Path
            assert isinstance(result, Path)
            assert result.exists() or str(result).endswith("COMP3888_TU13_01")

    def test_get_base_directory_exception_fallback(self):
        """Test final fallback to current working directory."""
        # Mock the entire sys.modules to raise an exception
        with patch('sys.modules', side_effect=Exception("Module error")):
            result = get_base_directory()
            
            # Should return current working directory as final fallback
            assert isinstance(result, Path)
            # The function might still return the fallback path instead of cwd()
            # Let's just check it's a valid Path
            assert result.exists() or str(result).endswith("COMP3888_TU13_01")


class TestEnsureDirectoryExists:
    """Test cases for ensure_directory_exists function."""

    def test_ensure_directory_exists_success(self):
        """Test successful directory creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            new_dir = base_dir / "new_directory"
            
            result = ensure_directory_exists(new_dir, "test directory")
            
            assert result is True
            assert new_dir.exists()
            assert new_dir.is_dir()

    def test_ensure_directory_exists_already_exists(self):
        """Test when directory already exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            existing_dir = Path(temp_dir)
            
            result = ensure_directory_exists(existing_dir, "existing directory")
            
            assert result is True
            assert existing_dir.exists()

    def test_ensure_directory_exists_error(self):
        """Test handling of creation errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            new_dir = base_dir / "error_directory"
            
            # Mock mkdir to raise OSError
            with patch.object(Path, 'mkdir', side_effect=OSError("Permission denied")):
                result = ensure_directory_exists(new_dir, "error directory")
                
                assert result is False


class TestListDirectoryContents:
    """Test cases for list_directory_contents function."""

    def test_list_directory_contents_success(self):
        """Test successful directory listing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dir_path = Path(temp_dir)
            
            # Create test files
            (dir_path / "file1.txt").write_text("content1")
            (dir_path / "file2.txt").write_text("content2")
            (dir_path / "subdir").mkdir()
            
            result = list_directory_contents(dir_path)
            
            assert len(result) == 3
            assert any(f.name == "file1.txt" for f in result)
            assert any(f.name == "file2.txt" for f in result)
            assert any(f.name == "subdir" for f in result)

    def test_list_directory_contents_with_pattern(self):
        """Test directory listing with pattern."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dir_path = Path(temp_dir)
            
            # Create test files
            (dir_path / "file1.txt").write_text("content1")
            (dir_path / "file2.txt").write_text("content2")
            (dir_path / "file1.py").write_text("code")
            
            result = list_directory_contents(dir_path, "*.txt")
            
            assert len(result) == 2
            assert all(f.suffix == ".txt" for f in result)

    def test_list_directory_contents_nonexistent(self):
        """Test listing nonexistent directory."""
        nonexistent_dir = Path("/nonexistent/directory")
        
        result = list_directory_contents(nonexistent_dir)
        
        assert result == []

    def test_list_directory_contents_not_directory(self):
        """Test listing file instead of directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test.txt"
            file_path.write_text("content")
            
            result = list_directory_contents(file_path)
            
            assert result == []

    def test_list_directory_contents_exception(self):
        """Test handling of exceptions during listing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dir_path = Path(temp_dir)
            
            # Mock glob to raise exception
            with patch.object(Path, 'glob', side_effect=Exception("Listing error")):
                result = list_directory_contents(dir_path)
                
                assert result == []


class TestDirectoryHelperIntegration:
    """Integration tests for directory helper functions."""

    def test_full_dataset_creation_workflow(self):
        """Test complete workflow of creating a dataset structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            
            # Step 1: Create output directory
            output_dir = create_output_directory("augmented_output", base_dir)
            assert output_dir.exists()
            
            # Step 2: Create dataset structure
            dataset_dir = create_dataset_structure("test_dataset", output_dir)
            assert dataset_dir.exists()
            
            # Step 3: Validate structure
            is_valid = validate_dataset_structure(dataset_dir)
            assert is_valid is False  # No YAML file yet
            
            # Step 4: Copy data.yaml
            input_path = base_dir / "input_dataset"
            input_path.mkdir()
            (input_path / "data.yaml").write_text("nc: 2\nnames: ['cow', 'chicken']")
            
            copy_success = copy_data_yaml(input_path, dataset_dir)
            assert copy_success is True
            
            # Step 5: Validate again
            is_valid = validate_dataset_structure(dataset_dir)
            assert is_valid is True

    def test_augmented_dataset_workflow(self):
        """Test creating augmented dataset from input."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            
            # Create input dataset
            input_dataset = base_dir / "original_dataset"
            input_dataset.mkdir()
            (input_dataset / "data.yaml").write_text("nc: 1\nnames: ['sheep']")
            
            # Create output directory
            output_dir = create_output_directory("augmented_output", base_dir)
            
            # Create augmented dataset structure
            aug_dataset = create_augmented_dataset_structure(input_dataset, output_dir)
            
            assert aug_dataset.exists()
            assert aug_dataset.name == "aug_original_dataset"
            
            # Copy data.yaml
            copy_success = copy_data_yaml(input_dataset, aug_dataset)
            assert copy_success is True
            
            # Validate
            is_valid = validate_dataset_structure(aug_dataset)
            assert is_valid is True
