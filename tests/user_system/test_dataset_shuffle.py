"""
Test suite for user_system.utils.dataset_shuffle module.

This module tests the dataset shuffling and compilation functionality,
including file operations, directory management, and subprocess execution.
"""

import sys
import json
import tempfile
import subprocess
import threading
import queue
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import pytest
import yaml
import hashlib
import shutil

# Put repo root on sys.path
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Import the module under test
from user_system.utils.dataset_shuffle import (
    load_aliases_from_yaml,
    ensure_dir,
    DirCache,
    sha256_of_file,
    quick_file_signature,
    fast_validate_label,
    link_or_copy,
    safe_stem,
    unique_name,
    iter_images,
    candidate_split_image_dirs,
    label_for,
    collect_pairs,
    copy_pair,
    split_counts,
    copy_non_data_tree,
    reshuffle_datasets
)


class TestLoadAliasesFromYaml:
    """Test cases for load_aliases_from_yaml function."""

    def test_load_aliases_from_yaml_success(self):
        """Test successful loading of aliases from YAML."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            yaml_file = temp_path / "flag_helper.yaml"
            
            yaml_content = """
aliases:
  cow:
    - cattle
    - COW
    - bovine
  chicken:
    - hen
    - rooster
"""
            yaml_file.write_text(yaml_content)
            
            result = load_aliases_from_yaml(yaml_file)
            
            expected = {
                "cattle": "cow",
                "cow": "cow", 
                "bovine": "cow",
                "hen": "chicken",
                "rooster": "chicken"
            }
            assert result == expected

    def test_load_aliases_from_yaml_file_not_found(self):
        """Test when YAML file doesn't exist."""
        temp_path = Path("/nonexistent/path/flag_helper.yaml")
        
        result = load_aliases_from_yaml(temp_path)
        
        assert result == {}

    def test_load_aliases_from_yaml_no_aliases(self):
        """Test when YAML file has no aliases section."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            yaml_file = temp_path / "flag_helper.yaml"
            
            yaml_content = """
other_config:
  value: 123
"""
            yaml_file.write_text(yaml_content)
            
            result = load_aliases_from_yaml(yaml_file)
            
            assert result == {}

    def test_load_aliases_from_yaml_invalid_yaml(self):
        """Test when YAML file is invalid."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            yaml_file = temp_path / "flag_helper.yaml"
            
            yaml_file.write_text("invalid: yaml: content: [")
            
            result = load_aliases_from_yaml(yaml_file)
            
            assert result == {}

    def test_load_aliases_from_yaml_empty_aliases(self):
        """Test when aliases section is empty."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            yaml_file = temp_path / "flag_helper.yaml"
            
            yaml_content = """
aliases: {}
"""
            yaml_file.write_text(yaml_content)
            
            result = load_aliases_from_yaml(yaml_file)
            
            assert result == {}


class TestEnsureDir:
    """Test cases for ensure_dir function."""

    def test_ensure_dir_new_directory(self):
        """Test creating a new directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            new_dir = temp_path / "new_directory"
            
            ensure_dir(new_dir)
            
            assert new_dir.exists()
            assert new_dir.is_dir()

    def test_ensure_dir_existing_directory(self):
        """Test when directory already exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            existing_dir = temp_path / "existing_directory"
            existing_dir.mkdir()
            
            # Should not raise an exception
            ensure_dir(existing_dir)
            
            assert existing_dir.exists()
            assert existing_dir.is_dir()

    def test_ensure_dir_nested_directories(self):
        """Test creating nested directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            nested_dir = temp_path / "level1" / "level2" / "level3"
            
            ensure_dir(nested_dir)
            
            assert nested_dir.exists()
            assert nested_dir.is_dir()


class TestDirCache:
    """Test cases for DirCache class."""

    def test_dir_cache_ensure_new_directory(self):
        """Test creating a new directory with DirCache."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            new_dir = temp_path / "new_directory"
            
            cache = DirCache()
            cache.ensure(new_dir)
            
            assert new_dir.exists()
            assert new_dir.is_dir()

    def test_dir_cache_ensure_existing_directory(self):
        """Test ensuring an existing directory with DirCache."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            existing_dir = temp_path / "existing_directory"
            existing_dir.mkdir()
            
            cache = DirCache()
            cache.ensure(existing_dir)
            
            assert existing_dir.exists()
            assert existing_dir.is_dir()

    def test_dir_cache_thread_safety(self):
        """Test DirCache thread safety."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            cache = DirCache()
            
            def create_dir(i):
                new_dir = temp_path / f"dir_{i}"
                cache.ensure(new_dir)
                return new_dir
            
            # Create directories from multiple threads
            threads = []
            results = []
            
            for i in range(5):
                thread = threading.Thread(target=lambda i=i: results.append(create_dir(i)))
                threads.append(thread)
                thread.start()
            
            for thread in threads:
                thread.join()
            
            # All directories should exist
            for result in results:
                assert result.exists()
                assert result.is_dir()


class TestSha256OfFile:
    """Test cases for sha256_of_file function."""

    def test_sha256_of_file(self):
        """Test SHA256 calculation of a file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_file = temp_path / "test.txt"
            test_content = "Hello, World!"
            test_file.write_text(test_content)
            
            result = sha256_of_file(test_file)
            
            # Calculate expected SHA256
            expected = hashlib.sha256(test_content.encode()).hexdigest()
            assert result == expected

    def test_sha256_of_file_empty(self):
        """Test SHA256 calculation of an empty file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_file = temp_path / "empty.txt"
            test_file.write_text("")
            
            result = sha256_of_file(test_file)
            
            # SHA256 of empty string
            expected = hashlib.sha256(b"").hexdigest()
            assert result == expected

    def test_sha256_of_file_large(self):
        """Test SHA256 calculation of a large file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_file = temp_path / "large.txt"
            
            # Create a large file
            large_content = "x" * (1024 * 1024)  # 1MB
            test_file.write_text(large_content)
            
            result = sha256_of_file(test_file)
            
            # Calculate expected SHA256
            expected = hashlib.sha256(large_content.encode()).hexdigest()
            assert result == expected


class TestQuickFileSignature:
    """Test cases for quick_file_signature function."""

    def test_quick_file_signature(self):
        """Test quick file signature calculation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_file = temp_path / "test.txt"
            test_content = "Hello, World!"
            test_file.write_text(test_content)
            
            result = quick_file_signature(test_file)
            
            assert len(result) == 3
            assert isinstance(result[0], int)  # size
            assert isinstance(result[1], int)  # mtime_ns
            assert isinstance(result[2], str)  # hash

    def test_quick_file_signature_nonexistent_file(self):
        """Test quick file signature of nonexistent file."""
        temp_path = Path("/nonexistent/file.txt")
        
        result = quick_file_signature(temp_path)
        
        assert result == (-1, 0, "")


class TestFastValidateLabel:
    """Test cases for fast_validate_label function."""

    def test_fast_validate_label_valid(self):
        """Test validation of a valid label file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            label_file = temp_path / "label.txt"
            label_file.write_text("0 0.5 0.5 0.2 0.3\n")
            
            valid, class_id = fast_validate_label(label_file)
            
            assert valid is True
            assert class_id == 0

    def test_fast_validate_label_empty_file(self):
        """Test validation of an empty label file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            label_file = temp_path / "empty.txt"
            label_file.write_text("")
            
            valid, class_id = fast_validate_label(label_file)
            
            assert valid is True
            assert class_id == -1

    def test_fast_validate_label_nonexistent_file(self):
        """Test validation of nonexistent label file."""
        temp_path = Path("/nonexistent/label.txt")
        
        valid, class_id = fast_validate_label(temp_path)
        
        assert valid is True
        assert class_id == -1

    def test_fast_validate_label_invalid_format(self):
        """Test validation of invalid label format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            label_file = temp_path / "invalid.txt"
            label_file.write_text("invalid format\n")
            
            valid, class_id = fast_validate_label(label_file)
            
            assert valid is False
            assert class_id == -1

    def test_fast_validate_label_with_empty_lines(self):
        """Test validation of label file with empty lines."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            label_file = temp_path / "label.txt"
            label_file.write_text("\n\n0 0.5 0.5 0.2 0.3\n\n")
            
            valid, class_id = fast_validate_label(label_file)
            
            assert valid is True
            assert class_id == 0


class TestLinkOrCopy:
    """Test cases for link_or_copy function."""

    def test_link_or_copy_success(self):
        """Test successful linking or copying."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            src_file = temp_path / "source.txt"
            dst_file = temp_path / "destination.txt"
            
            src_file.write_text("Test content")
            
            link_or_copy(src_file, dst_file)
            
            assert dst_file.exists()
            assert dst_file.read_text() == "Test content"

    def test_link_or_copy_with_copy(self):
        """Test copying when linking fails."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            src_file = temp_path / "source.txt"
            dst_file = temp_path / "destination.txt"
            
            src_file.write_text("Test content")
            
            # Mock os.link and os.symlink to fail
            with patch('os.link', side_effect=OSError("Link failed")), \
                 patch('os.symlink', side_effect=OSError("Symlink failed")):
                
                link_or_copy(src_file, dst_file)
            
            assert dst_file.exists()
            assert dst_file.read_text() == "Test content"


class TestSafeStem:
    """Test cases for safe_stem function."""

    def test_safe_stem_normal(self):
        """Test safe_stem with normal input."""
        result = safe_stem("image_001")
        assert result == "image_001"

    def test_safe_stem_long(self):
        """Test safe_stem with long input."""
        long_stem = "a" * 50
        result = safe_stem(long_stem)
        assert len(result) <= 40
        assert result == "a" * 40

    def test_safe_stem_empty(self):
        """Test safe_stem with empty input."""
        result = safe_stem("")
        assert result == "img"

    def test_safe_stem_none(self):
        """Test safe_stem with None input."""
        result = safe_stem(None)
        assert result == "img"

    def test_safe_stem_with_special_chars(self):
        """Test safe_stem with special characters."""
        result = safe_stem("image.001_-")
        assert result == "image.001"


class TestUniqueName:
    """Test cases for unique_name function."""

    def test_unique_name(self):
        """Test unique name generation."""
        result = unique_name("image", ".jpg", "abc12345")
        assert result == "image__abc12345.jpg"

    def test_unique_name_long_hash(self):
        """Test unique name with long hash."""
        result = unique_name("image", ".jpg", "abcdefghijklmnop")
        assert result == "image__abcdefgh.jpg"


class TestIterImages:
    """Test cases for iter_images function."""

    def test_iter_images_success(self):
        """Test iterating over images in directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test images
            (temp_path / "image1.jpg").write_text("fake image")
            (temp_path / "image2.png").write_text("fake image")
            (temp_path / "text.txt").write_text("not an image")
            
            images = list(iter_images(temp_path))
            
            assert len(images) == 2
            assert any("image1.jpg" in str(img) for img in images)
            assert any("image2.png" in str(img) for img in images)

    def test_iter_images_nonexistent_directory(self):
        """Test iterating over nonexistent directory."""
        temp_path = Path("/nonexistent/directory")
        
        images = list(iter_images(temp_path))
        
        assert len(images) == 0

    def test_iter_images_recursive(self):
        """Test recursive image iteration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create nested structure
            subdir = temp_path / "subdir"
            subdir.mkdir()
            
            (temp_path / "image1.jpg").write_text("fake image")
            (subdir / "image2.png").write_text("fake image")
            
            images = list(iter_images(temp_path))
            
            assert len(images) == 2
            assert any("image1.jpg" in str(img) for img in images)
            assert any("image2.png" in str(img) for img in images)


class TestCandidateSplitImageDirs:
    """Test cases for candidate_split_image_dirs function."""

    def test_candidate_split_image_dirs_train(self):
        """Test finding train image directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create train directory
            train_dir = temp_path / "images" / "train"
            train_dir.mkdir(parents=True)
            
            result = candidate_split_image_dirs(temp_path, "train")
            
            assert len(result) == 1
            assert result[0] == train_dir

    def test_candidate_split_image_dirs_val(self):
        """Test finding validation image directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create val directory
            val_dir = temp_path / "images" / "val"
            val_dir.mkdir(parents=True)
            
            result = candidate_split_image_dirs(temp_path, "val")
            
            assert len(result) == 1
            assert result[0] == val_dir

    def test_candidate_split_image_dirs_valid_alias(self):
        """Test finding validation directories with 'valid' alias."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create valid directory
            valid_dir = temp_path / "images" / "valid"
            valid_dir.mkdir(parents=True)
            
            result = candidate_split_image_dirs(temp_path, "val")
            
            assert len(result) == 1
            assert result[0] == valid_dir

    def test_candidate_split_image_dirs_test_variants(self):
        """Test finding test directories with variants."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test variants
            test_dir = temp_path / "images" / "test"
            test_dir.mkdir(parents=True)
            test_custom_dir = temp_path / "images" / "test_custom"
            test_custom_dir.mkdir(parents=True)
            
            result = candidate_split_image_dirs(temp_path, "test")
            
            assert len(result) == 2
            assert any("test" in str(path) for path in result)
            assert any("test_custom" in str(path) for path in result)

    def test_candidate_split_image_dirs_no_directories(self):
        """Test when no directories exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            result = candidate_split_image_dirs(temp_path, "train")
            
            assert len(result) == 0


class TestLabelFor:
    """Test cases for label_for function."""

    def test_label_for_success(self):
        """Test finding label for image."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create image and label structure
            img_dir = temp_path / "images" / "train"
            lbl_dir = temp_path / "labels" / "train"
            img_dir.mkdir(parents=True)
            lbl_dir.mkdir(parents=True)
            
            img_file = img_dir / "image.jpg"
            lbl_file = lbl_dir / "image.txt"
            img_file.write_text("fake image")
            lbl_file.write_text("0 0.5 0.5 0.2 0.3")
            
            result = label_for(img_file)
            
            assert result == lbl_file

    def test_label_for_no_images_root(self):
        """Test when no images root is found."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            img_file = temp_path / "image.jpg"
            img_file.write_text("fake image")
            
            result = label_for(img_file)
            
            assert result is None

    def test_label_for_no_label_file(self):
        """Test when label file doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create image structure without label
            img_dir = temp_path / "images" / "train"
            img_dir.mkdir(parents=True)
            
            img_file = img_dir / "image.jpg"
            img_file.write_text("fake image")
            
            result = label_for(img_file)
            
            assert result is None


class TestCollectPairs:
    """Test cases for collect_pairs function."""

    def test_collect_pairs_standard_structure(self):
        """Test collecting pairs from standard dataset structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create standard structure
            for split in ["train", "val", "test"]:
                img_dir = temp_path / "images" / split
                lbl_dir = temp_path / "labels" / split
                img_dir.mkdir(parents=True)
                lbl_dir.mkdir(parents=True)
                
                img_file = img_dir / f"image_{split}.jpg"
                lbl_file = lbl_dir / f"image_{split}.txt"
                img_file.write_text("fake image")
                lbl_file.write_text("0 0.5 0.5 0.2 0.3")
            
            pairs = collect_pairs(temp_path)
            
            assert len(pairs) == 3
            assert all(isinstance(pair, tuple) and len(pair) == 2 for pair in pairs)

    def test_collect_pairs_flat_structure(self):
        """Test collecting pairs from flat structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create flat structure
            img_dir = temp_path / "images"
            lbl_dir = temp_path / "labels"
            img_dir.mkdir(parents=True)
            lbl_dir.mkdir(parents=True)
            
            img_file = img_dir / "image.jpg"
            lbl_file = lbl_dir / "image.txt"
            img_file.write_text("fake image")
            lbl_file.write_text("0 0.5 0.5 0.2 0.3")
            
            pairs = collect_pairs(temp_path)
            
            assert len(pairs) == 1
            assert pairs[0][0] == img_file
            assert pairs[0][1] == lbl_file

    def test_collect_pairs_no_images(self):
        """Test collecting pairs when no images exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            pairs = collect_pairs(temp_path)
            
            assert len(pairs) == 0


class TestCopyPair:
    """Test cases for copy_pair function."""

    def test_copy_pair_success(self):
        """Test successful copying of image-label pair."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create source files
            src_img = temp_path / "source.jpg"
            src_lbl = temp_path / "source.txt"
            src_img.write_text("fake image content")
            src_lbl.write_text("0 0.5 0.5 0.2 0.3")
            
            out_root = temp_path / "output"
            
            out_img, out_lbl, hash_val = copy_pair(src_img, src_lbl, out_root, "train")
            
            assert out_img.exists()
            assert out_lbl.exists()
            assert out_img.read_text() == "fake image content"
            assert out_lbl.read_text() == "0 0.5 0.5 0.2 0.3"
            assert isinstance(hash_val, str)
            assert len(hash_val) == 64  # SHA256 hex length

    def test_copy_pair_with_dir_cache(self):
        """Test copying pair with directory cache."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create source files
            src_img = temp_path / "source.jpg"
            src_lbl = temp_path / "source.txt"
            src_img.write_text("fake image content")
            src_lbl.write_text("0 0.5 0.5 0.2 0.3")
            
            out_root = temp_path / "output"
            dir_cache = DirCache()
            
            out_img, out_lbl, hash_val = copy_pair(src_img, src_lbl, out_root, "train", dir_cache)
            
            assert out_img.exists()
            assert out_lbl.exists()
            assert isinstance(hash_val, str)


class TestSplitCounts:
    """Test cases for split_counts function."""

    def test_split_counts_normal(self):
        """Test normal split counts calculation."""
        train, val, test = split_counts(100, (0.8, 0.1, 0.1))
        
        assert train == 81  # 100 - 10 - 9 = 81 (remainder goes to train)
        assert val == 10
        assert test == 9
        assert train + val + test == 100

    def test_split_counts_remainder_to_train(self):
        """Test that remainder goes to train."""
        train, val, test = split_counts(101, (0.8, 0.1, 0.1))
        
        assert train == 81  # 80 + 1 remainder
        assert val == 10
        assert test == 10
        assert train + val + test == 101

    def test_split_counts_custom_ratios(self):
        """Test with custom ratios."""
        train, val, test = split_counts(100, (0.7, 0.2, 0.1))
        
        assert train == 71  # 100 - 20 - 9 = 71 (remainder goes to train)
        assert val == 20
        assert test == 9
        assert train + val + test == 100

    def test_split_counts_small_number(self):
        """Test with small number."""
        train, val, test = split_counts(3, (0.8, 0.1, 0.1))
        
        assert train == 3  # 3 - 0 - 0 = 3 (all goes to train)
        assert val == 0
        assert test == 0
        assert train + val + test == 3


class TestCopyNonDataTree:
    """Test cases for copy_non_data_tree function."""

    def test_copy_non_data_tree_success(self):
        """Test successful copying excluding images/labels."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create source structure
            src = temp_path / "source"
            src.mkdir()
            
            # Create files to copy
            (src / "data.yaml").write_text("test: data")
            (src / "README.txt").write_text("test readme")
            
            # Create images/labels to exclude
            (src / "images").mkdir()
            (src / "labels").mkdir()
            (src / "images" / "image.jpg").write_text("fake image")
            (src / "labels" / "label.txt").write_text("fake label")
            
            dst = temp_path / "destination"
            
            copy_non_data_tree(src, dst)
            
            # Check that non-data files were copied
            assert (dst / "data.yaml").exists()
            assert (dst / "README.txt").exists()
            
            # Check that images/labels were excluded
            assert not (dst / "images").exists()
            assert not (dst / "labels").exists()

    def test_copy_non_data_tree_existing_dst(self):
        """Test copying when destination exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create source structure
            src = temp_path / "source"
            src.mkdir()
            (src / "data.yaml").write_text("test: data")
            
            # Create existing destination
            dst = temp_path / "destination"
            dst.mkdir()
            (dst / "existing.txt").write_text("existing content")
            
            copy_non_data_tree(src, dst)
            
            # Destination should be recreated
            assert (dst / "data.yaml").exists()
            assert not (dst / "existing.txt").exists()


class TestReshuffleDatasets:
    """Test cases for reshuffle_datasets function."""

    def test_reshuffle_datasets_success(self):
        """Test successful dataset reshuffling."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create source dataset
            src_dataset = temp_path / "source_dataset"
            src_dataset.mkdir()
            
            # Create standard structure
            for split in ["train", "val", "test"]:
                img_dir = src_dataset / "images" / split
                lbl_dir = src_dataset / "labels" / split
                img_dir.mkdir(parents=True)
                lbl_dir.mkdir(parents=True)
                
                img_file = img_dir / f"image_{split}.jpg"
                lbl_file = lbl_dir / f"image_{split}.txt"
                img_file.write_text("fake image content")
                lbl_file.write_text("0 0.5 0.5 0.2 0.3")
            
            out_dir = temp_path / "output"
            
            with patch('user_system.utils.dataset_shuffle.subprocess.run') as mock_run:
                mock_proc = MagicMock()
                mock_proc.returncode = 0
                mock_proc.stdout = "Compilation successful"
                mock_run.return_value = mock_proc
                
                result = reshuffle_datasets([src_dataset], out_dir, seed=42)
                
                assert isinstance(result, dict)
                assert "source_dataset" in result
                assert "_compiled_out" in result
                mock_run.assert_called_once()

    def test_reshuffle_datasets_nonexistent_root(self):
        """Test reshuffling with nonexistent dataset root."""
        temp_path = Path("/nonexistent/dataset")
        
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "out"
            with pytest.raises(FileNotFoundError) as exc_info:
                reshuffle_datasets([temp_path], out_dir)
        assert "Dataset root not found" in str(exc_info.value)

    def test_reshuffle_datasets_compile_failure(self):
        """Test when compile.py fails."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create source dataset
            src_dataset = temp_path / "source_dataset"
            src_dataset.mkdir()
            
            # Create minimal structure
            img_dir = src_dataset / "images" / "train"
            lbl_dir = src_dataset / "labels" / "train"
            img_dir.mkdir(parents=True)
            lbl_dir.mkdir(parents=True)
            
            img_file = img_dir / "image.jpg"
            lbl_file = lbl_dir / "image.txt"
            img_file.write_text("fake image content")
            lbl_file.write_text("0 0.5 0.5 0.2 0.3")
            
            out_dir = temp_path / "output"
            
            with patch('user_system.utils.dataset_shuffle.subprocess.run') as mock_run:
                mock_proc = MagicMock()
                mock_proc.returncode = 1
                mock_proc.stdout = "Error output"
                mock_proc.stderr = "Error message"
                mock_run.return_value = mock_proc
                
                with pytest.raises(RuntimeError) as exc_info:
                    reshuffle_datasets([src_dataset], out_dir)
                assert "compile.py failed" in str(exc_info.value)

    def test_reshuffle_datasets_with_aliases(self):
        """Test reshuffling with aliases configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create source dataset
            src_dataset = temp_path / "source_dataset"
            src_dataset.mkdir()
            
            # Create minimal structure
            img_dir = src_dataset / "images" / "train"
            lbl_dir = src_dataset / "labels" / "train"
            img_dir.mkdir(parents=True)
            lbl_dir.mkdir(parents=True)
            
            img_file = img_dir / "image.jpg"
            lbl_file = lbl_dir / "image.txt"
            img_file.write_text("fake image content")
            lbl_file.write_text("0 0.5 0.5 0.2 0.3")
            
            # Create flag_helper.yaml with aliases
            config_dir = temp_path / "configs"
            config_dir.mkdir()
            flag_file = config_dir / "flag_helper.yaml"
            flag_file.write_text("""
aliases:
  cow: [cattle, COW]
keep_classes: [0, 1]
""")
            
            out_dir = temp_path / "output"
            
            with patch('user_system.utils.dataset_shuffle.subprocess.run') as mock_run, \
                 patch('user_system.utils.dataset_shuffle.Path') as mock_path:
                
                # Mock the flag file path resolution
                mock_path.side_effect = lambda x: flag_file if "flag_helper.yaml" in str(x) else Path(x)
                
                mock_proc = MagicMock()
                mock_proc.returncode = 0
                mock_proc.stdout = "Compilation successful"
                mock_run.return_value = mock_proc
                
                result = reshuffle_datasets([src_dataset], out_dir)
                
                assert isinstance(result, dict)
                mock_run.assert_called_once()

    def test_reshuffle_datasets_multiple_datasets(self):
        """Test reshuffling multiple datasets."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create multiple source datasets
            datasets = []
            for i in range(2):
                src_dataset = temp_path / f"dataset_{i}"
                src_dataset.mkdir()
                
                # Create minimal structure
                img_dir = src_dataset / "images" / "train"
                lbl_dir = src_dataset / "labels" / "train"
                img_dir.mkdir(parents=True)
                lbl_dir.mkdir(parents=True)
                
                img_file = img_dir / "image.jpg"
                lbl_file = lbl_dir / "image.txt"
                img_file.write_text("fake image content")
                lbl_file.write_text("0 0.5 0.5 0.2 0.3")
                
                datasets.append(src_dataset)
            
            out_dir = temp_path / "output"
            
            with patch('user_system.utils.dataset_shuffle.subprocess.run') as mock_run:
                mock_proc = MagicMock()
                mock_proc.returncode = 0
                mock_proc.stdout = "Compilation successful"
                mock_run.return_value = mock_proc
                
                result = reshuffle_datasets(datasets, out_dir)
                
                assert isinstance(result, dict)
                assert "dataset_0" in result
                assert "dataset_1" in result
                assert "_compiled_out" in result
                mock_run.assert_called_once()

    def test_reshuffle_datasets_with_names(self):
        """Test reshuffling with names parameter."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create source dataset
            src_dataset = temp_path / "source_dataset"
            src_dataset.mkdir()
            
            # Create minimal structure
            img_dir = src_dataset / "images" / "train"
            lbl_dir = src_dataset / "labels" / "train"
            img_dir.mkdir(parents=True)
            lbl_dir.mkdir(parents=True)
            
            img_file = img_dir / "image.jpg"
            lbl_file = lbl_dir / "image.txt"
            img_file.write_text("fake image content")
            lbl_file.write_text("0 0.5 0.5 0.2 0.3")
            
            out_dir = temp_path / "output"
            names = ["class1", "class2"]
            
            with patch('user_system.utils.dataset_shuffle.subprocess.run') as mock_run:
                mock_proc = MagicMock()
                mock_proc.returncode = 0
                mock_proc.stdout = "Compilation successful"
                mock_run.return_value = mock_proc
                
                result = reshuffle_datasets([src_dataset], out_dir, names=names)
                
                assert isinstance(result, dict)
                mock_run.assert_called_once()
