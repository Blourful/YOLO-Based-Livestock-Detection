"""
Test suite for user_system.utils.modify_helper module.

This module tests the dataset label modification functionality
including YAML handling, label file updates, and dataset validation.
"""

import sys
import tempfile
import shutil
import yaml
from pathlib import Path
from unittest.mock import patch, mock_open
import pytest

# Put repo root on sys.path
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Import the module under test
from user_system.utils.modify_helper import (
    load_dataset_yaml,
    save_dataset_yaml,
    find_label_files,
    update_label_file,
    modify_dataset_labels,
    validate_dataset_structure
)


class TestLoadDatasetYaml:
    """Test cases for load_dataset_yaml function."""

    def test_load_dataset_yaml_success(self):
        """Test successful loading of dataset.yaml."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            yaml_file = temp_path / "dataset.yaml"
            
            yaml_content = """
train: images/train
val: images/val
test: images/test
nc: 2
names: ['cow', 'chicken']
"""
            yaml_file.write_text(yaml_content)
            
            result = load_dataset_yaml(temp_path)
            
            assert result["nc"] == 2
            assert result["names"] == ['cow', 'chicken']
            assert result["train"] == "images/train"

    def test_load_dataset_yaml_data_yaml(self):
        """Test loading data.yaml when dataset.yaml doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            yaml_file = temp_path / "data.yaml"
            
            yaml_content = """
train: images/train
val: images/val
nc: 1
names: ['sheep']
"""
            yaml_file.write_text(yaml_content)
            
            result = load_dataset_yaml(temp_path)
            
            assert result["nc"] == 1
            assert result["names"] == ['sheep']

    def test_load_dataset_yaml_priority_order(self):
        """Test that dataset.yaml takes priority over data.yaml."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create both files
            dataset_yaml = temp_path / "dataset.yaml"
            data_yaml = temp_path / "data.yaml"
            
            dataset_yaml.write_text("nc: 2\nnames: ['cow', 'chicken']")
            data_yaml.write_text("nc: 1\nnames: ['sheep']")
            
            result = load_dataset_yaml(temp_path)
            
            # Should load dataset.yaml (priority)
            assert result["nc"] == 2
            assert result["names"] == ['cow', 'chicken']

    def test_load_dataset_yaml_invalid_yaml(self):
        """Test handling of invalid YAML content."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            yaml_file = temp_path / "dataset.yaml"
            
            yaml_file.write_text("invalid: yaml: content: [")
            
            with pytest.raises(FileNotFoundError):
                load_dataset_yaml(temp_path)

    def test_load_dataset_yaml_empty_yaml(self):
        """Test handling of empty YAML file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            yaml_file = temp_path / "dataset.yaml"
            
            yaml_file.write_text("")
            
            result = load_dataset_yaml(temp_path)
            
            assert result == {}

    def test_load_dataset_yaml_no_files(self):
        """Test when no YAML files exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            with pytest.raises(FileNotFoundError):
                load_dataset_yaml(temp_path)


class TestSaveDatasetYaml:
    """Test cases for save_dataset_yaml function."""

    def test_save_dataset_yaml_success(self):
        """Test successful saving of dataset.yaml."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            data = {
                "train": "images/train",
                "val": "images/val",
                "nc": 2,
                "names": ['cow', 'chicken']
            }
            
            save_dataset_yaml(temp_path, data)
            
            yaml_file = temp_path / "dataset.yaml"
            assert yaml_file.exists()
            
            with open(yaml_file, 'r') as f:
                loaded_data = yaml.safe_load(f)
            
            assert loaded_data == data

    def test_save_dataset_yaml_overwrite(self):
        """Test overwriting existing dataset.yaml."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create initial file
            initial_data = {"nc": 1, "names": ['sheep']}
            save_dataset_yaml(temp_path, initial_data)
            
            # Overwrite with new data
            new_data = {"nc": 2, "names": ['cow', 'chicken']}
            save_dataset_yaml(temp_path, new_data)
            
            yaml_file = temp_path / "dataset.yaml"
            with open(yaml_file, 'r') as f:
                loaded_data = yaml.safe_load(f)
            
            assert loaded_data == new_data


class TestFindLabelFiles:
    """Test cases for find_label_files function."""

    def test_find_label_files_standard_structure(self):
        """Test finding label files in standard YOLO structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create standard structure
            for split in ["train", "val", "test"]:
                labels_dir = temp_path / "labels" / split
                labels_dir.mkdir(parents=True)
                (labels_dir / f"image_{split}.txt").write_text("0 0.5 0.5 0.2 0.3")
            
            result = find_label_files(temp_path)
            
            assert len(result) == 3
            assert all(f.name.endswith('.txt') for f in result)

    def test_find_label_files_split_labels_structure(self):
        """Test finding label files in split/labels structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create split/labels structure
            for split in ["train", "val"]:
                labels_dir = temp_path / split / "labels"
                labels_dir.mkdir(parents=True)
                (labels_dir / f"image_{split}.txt").write_text("0 0.5 0.5 0.2 0.3")
            
            result = find_label_files(temp_path)
            
            assert len(result) == 2
            assert all(f.name.endswith('.txt') for f in result)

    def test_find_label_files_mixed_structure(self):
        """Test finding label files in mixed structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create mixed structure
            labels_dir1 = temp_path / "labels" / "train"
            labels_dir1.mkdir(parents=True)
            (labels_dir1 / "image1.txt").write_text("0 0.5 0.5 0.2 0.3")
            
            labels_dir2 = temp_path / "val" / "labels"
            labels_dir2.mkdir(parents=True)
            (labels_dir2 / "image2.txt").write_text("1 0.3 0.7 0.1 0.2")
            
            result = find_label_files(temp_path)
            
            assert len(result) == 2

    def test_find_label_files_no_labels(self):
        """Test when no label files exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            result = find_label_files(temp_path)
            
            assert len(result) == 0

    def test_find_label_files_valid_alias(self):
        """Test finding label files with 'valid' alias."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create labels/valid directory
            labels_dir = temp_path / "labels" / "valid"
            labels_dir.mkdir(parents=True)
            (labels_dir / "image.txt").write_text("0 0.5 0.5 0.2 0.3")
            
            result = find_label_files(temp_path)
            
            assert len(result) == 1


class TestUpdateLabelFile:
    """Test cases for update_label_file function."""

    def test_update_label_file_success(self):
        """Test successful label file update."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            label_file = temp_path / "test.txt"
            
            # Create label file with class 0
            label_file.write_text("0 0.5 0.5 0.2 0.3\n1 0.3 0.7 0.1 0.2")
            
            label_mapping = {"0": "2", "1": "3"}
            
            result = update_label_file(label_file, "old_label", "new_label", label_mapping)
            
            assert result == 2  # 2 labels changed
            
            # Check updated content
            with open(label_file, 'r') as f:
                content = f.read()
            
            assert "2 0.5 0.5 0.2 0.3" in content
            assert "3 0.3 0.7 0.1 0.2" in content

    def test_update_label_file_no_changes(self):
        """Test label file update with no matching labels."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            label_file = temp_path / "test.txt"
            
            label_file.write_text("2 0.5 0.5 0.2 0.3\n3 0.3 0.7 0.1 0.2")
            
            label_mapping = {"0": "1"}  # No matching class IDs
            
            result = update_label_file(label_file, "old_label", "new_label", label_mapping)
            
            assert result == 0  # No labels changed

    def test_update_label_file_empty_file(self):
        """Test updating empty label file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            label_file = temp_path / "test.txt"
            
            label_file.write_text("")
            
            label_mapping = {"0": "1"}
            
            result = update_label_file(label_file, "old_label", "new_label", label_mapping)
            
            assert result == 0

    def test_update_label_file_invalid_format(self):
        """Test updating label file with invalid format lines."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            label_file = temp_path / "test.txt"
            
            label_file.write_text("0 0.5 0.5 0.2 0.3\ninvalid line\n1 0.3 0.7 0.1 0.2")
            
            label_mapping = {"0": "2", "1": "3"}
            
            result = update_label_file(label_file, "old_label", "new_label", label_mapping)
            
            assert result == 2  # Only valid lines changed

    def test_update_label_file_exception_handling(self):
        """Test exception handling in label file update."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            label_file = temp_path / "test.txt"
            
            # Create a file that will cause an exception when reading
            label_file.write_text("0 0.5 0.5 0.2 0.3")
            
            with patch('builtins.open', side_effect=IOError("File read error")):
                result = update_label_file(label_file, "old_label", "new_label", {"0": "1"})
                
                assert result == 0


class TestModifyDatasetLabels:
    """Test cases for modify_dataset_labels function."""

    def test_modify_dataset_labels_success(self):
        """Test successful dataset label modification."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create dataset structure
            dataset_yaml = temp_path / "dataset.yaml"
            dataset_yaml.write_text("""
train: images/train
val: images/val
nc: 2
names: ['cow', 'chicken']
""")
            
            # Create label files
            labels_dir = temp_path / "labels" / "train"
            labels_dir.mkdir(parents=True)
            (labels_dir / "image1.txt").write_text("0 0.5 0.5 0.2 0.3\n1 0.3 0.7 0.1 0.2")
            
            labels_dir = temp_path / "labels" / "val"
            labels_dir.mkdir(parents=True)
            (labels_dir / "image2.txt").write_text("0 0.4 0.6 0.15 0.25")
            
            output_path = temp_path / "output"
            
            result = modify_dataset_labels(temp_path, "cow", "cattle", output_path)
            
            assert result["files_processed"] == 2
            assert result["labels_changed"] == 2  # 2 labels with class 0
            assert result["classes_updated"] == 1  # Added new class
            
            # Check that output dataset was created
            assert output_path.exists()
            assert (output_path / "dataset.yaml").exists()

    def test_modify_dataset_labels_existing_new_label(self):
        """Test modification when new label already exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create dataset with both old and new labels
            dataset_yaml = temp_path / "dataset.yaml"
            dataset_yaml.write_text("""
train: images/train
val: images/val
nc: 3
names: ['cow', 'chicken', 'cattle']
""")
            
            # Create label files
            labels_dir = temp_path / "labels" / "train"
            labels_dir.mkdir(parents=True)
            (labels_dir / "image1.txt").write_text("0 0.5 0.5 0.2 0.3")  # class 0 (cow)
            
            output_path = temp_path / "output"
            
            result = modify_dataset_labels(temp_path, "cow", "cattle", output_path)
            
            assert result["files_processed"] == 1
            assert result["labels_changed"] == 1
            assert result["classes_updated"] == 0  # No new class added

    def test_modify_dataset_labels_label_not_found(self):
        """Test modification when old label doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            dataset_yaml = temp_path / "dataset.yaml"
            dataset_yaml.write_text("""
train: images/train
nc: 2
names: ['cow', 'chicken']
""")
            
            output_path = temp_path / "output"
            
            result = modify_dataset_labels(temp_path, "sheep", "cattle", output_path)
            
            assert result["files_processed"] == 0
            assert result["labels_changed"] == 0
            assert result["classes_updated"] == 0

    def test_modify_dataset_labels_no_yaml(self):
        """Test modification when no dataset YAML exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_path = temp_path / "output"
            
            with pytest.raises(FileNotFoundError):
                modify_dataset_labels(temp_path, "cow", "cattle", output_path)

    def test_modify_dataset_labels_same_path(self):
        """Test modification when output path is same as input."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            dataset_yaml = temp_path / "dataset.yaml"
            dataset_yaml.write_text("""
train: images/train
nc: 2
names: ['cow', 'chicken']
""")
            
            labels_dir = temp_path / "labels" / "train"
            labels_dir.mkdir(parents=True)
            (labels_dir / "image1.txt").write_text("0 0.5 0.5 0.2 0.3")
            
            result = modify_dataset_labels(temp_path, "cow", "cattle", temp_path)
            
            assert result["files_processed"] == 1
            assert result["labels_changed"] == 1

    def test_modify_dataset_labels_dict_names(self):
        """Test modification with dictionary-style names."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            dataset_yaml = temp_path / "dataset.yaml"
            dataset_yaml.write_text("""
train: images/train
nc: 2
names: {0: 'cow', 1: 'chicken'}
""")
            
            labels_dir = temp_path / "labels" / "train"
            labels_dir.mkdir(parents=True)
            (labels_dir / "image1.txt").write_text("0 0.5 0.5 0.2 0.3")
            
            output_path = temp_path / "output"
            
            result = modify_dataset_labels(temp_path, "cow", "cattle", output_path)
            
            # The function converts dict to list with class_0, class_1 format
            # So "cow" won't be found in the converted list
            assert result["files_processed"] == 0
            assert result["labels_changed"] == 0


class TestValidateDatasetStructure:
    """Test cases for validate_dataset_structure function."""

    def test_validate_dataset_structure_valid(self):
        """Test validation of valid dataset structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create valid dataset structure
            dataset_yaml = temp_path / "dataset.yaml"
            dataset_yaml.write_text("""
train: images/train
val: images/val
nc: 2
names: ['cow', 'chicken']
""")
            
            labels_dir = temp_path / "labels" / "train"
            labels_dir.mkdir(parents=True)
            (labels_dir / "image1.txt").write_text("0 0.5 0.5 0.2 0.3")
            
            result = validate_dataset_structure(temp_path)
            
            assert result is True

    def test_validate_dataset_structure_no_yaml(self):
        """Test validation when no YAML exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            result = validate_dataset_structure(temp_path)
            
            assert result is False

    def test_validate_dataset_structure_no_labels(self):
        """Test validation when no label files exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            dataset_yaml = temp_path / "dataset.yaml"
            dataset_yaml.write_text("""
train: images/train
nc: 2
names: ['cow', 'chicken']
""")
            
            result = validate_dataset_structure(temp_path)
            
            assert result is False

    def test_validate_dataset_structure_nonexistent_path(self):
        """Test validation of nonexistent path."""
        temp_path = Path("/nonexistent/path")
        
        result = validate_dataset_structure(temp_path)
        
        assert result is False

    def test_validate_dataset_structure_invalid_yaml(self):
        """Test validation with invalid YAML."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            dataset_yaml = temp_path / "dataset.yaml"
            dataset_yaml.write_text("invalid: yaml: content: [")
            
            result = validate_dataset_structure(temp_path)
            
            assert result is False
