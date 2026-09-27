"""
Directory Management Helper Module
=================================

This module provides utilities for creating and managing directory structures
for YOLO datasets, including output directories and dataset folder hierarchies.
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional
import sys


def create_output_directory(output_name: str, base_dir: Path) -> Path:
    """
    Create the main output directory for augmented datasets.
    
    Args:
        output_name: Name of the output directory (from config or default)
        base_dir: Base directory where main.py is located
        
    Returns:
        Path: Path to the created output directory
        
    Raises:
        OSError: If directory creation fails
    """
    output_path = base_dir / output_name
    
    try:
        # Create directory if it doesn't exist
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"📁 Output directory created/verified: {output_path}")
        return output_path
    except OSError as e:
        print(f"❌ Failed to create output directory: {output_path}")
        print(f"   Error: {e}")
        raise


def create_dataset_structure(dataset_name: str, output_dir: Path, subdirs: List[str] = None) -> Path:
    """
    Create a YOLO dataset directory structure.
    
    Args:
        dataset_name: Name of the dataset directory to create
        output_dir: Parent directory where the dataset structure will be created
        subdirs: List of subdirectories to create (defaults to ['train', 'test', 'valid'])
        
    Returns:
        Path: Path to the created dataset directory
        
    The structure created will be:
        output_dir/
        └── dataset_name/
            ├── train/
            │   ├── images/
            │   └── labels/
            ├── test/
            │   ├── images/
            │   └── labels/
            └── valid/
                ├── images/
                └── labels/
    """
    if subdirs is None:
        subdirs = ['train', 'test', 'valid']
    
    dataset_path = output_dir / dataset_name
    sub_subdirs = ['images', 'labels']
    
    print(f"🏗️  Creating dataset structure: {dataset_name}")
    
    try:
        # Create main dataset directory
        dataset_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories (train, test, valid)
        for subdir in subdirs:
            subdir_path = dataset_path / subdir
            subdir_path.mkdir(exist_ok=True)
            
            # Create images and labels subdirectories
            for sub_subdir in sub_subdirs:
                (subdir_path / sub_subdir).mkdir(exist_ok=True)
            
            print(f"   ✅ Created: {subdir}/images/ and {subdir}/labels/")
        
        return dataset_path
        
    except OSError as e:
        print(f"❌ Failed to create dataset structure for {dataset_name}")
        print(f"   Error: {e}")
        raise


def create_augmented_dataset_structure(input_path: Path, output_dir: Path) -> Path:
    """
    Create augmented dataset directory structure based on input dataset.
    
    Args:
        input_path: Path to the input dataset directory
        output_dir: Path to the main output directory
        
    Returns:
        Path: Path to the created augmented dataset directory
    """
    # Create augmented dataset name
    input_name = input_path.name
    aug_dataset_name = f"aug_{input_name}"
    
    return create_dataset_structure(aug_dataset_name, output_dir)


def copy_file(source_path: Path, dest_path: Path, description: str = "file") -> bool:
    """
    Copy a file from source to destination with error handling and logging.
    
    Args:
        source_path: Path to the source file
        dest_path: Path to the destination file
        description: Description of the file being copied (for logging)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        shutil.copy2(source_path, dest_path)
        print(f"   📋 Copied: {source_path.name} -> {dest_path.name}")
        return True
    except (OSError, shutil.Error) as e:
        print(f"❌ Failed to copy {description}: {source_path} to {dest_path}")
        print(f"   Error: {e}")
        return False


def copy_data_yaml(input_path: Path, dataset_path: Path) -> bool:
    """
    Copy data.yaml file from input dataset to target dataset directory.
    
    Args:
        input_path: Path to the input dataset directory
        dataset_path: Path to the target dataset directory
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Look for data.yaml or dataset.yaml in input directory
    yaml_candidates = ['data.yaml', 'dataset.yaml']
    source_yaml = None
    
    for candidate in yaml_candidates:
        candidate_path = input_path / candidate
        if candidate_path.exists():
            source_yaml = candidate_path
            break
    
    if source_yaml is None:
        print(f"⚠️  No data.yaml or dataset.yaml found in {input_path}")
        return False
    
    # Copy to target dataset directory as data.yaml
    dest_yaml = dataset_path / 'data.yaml'
    return copy_file(source_yaml, dest_yaml, "data.yaml")


def validate_dataset_structure(dataset_path: Path) -> bool:
    """
    Validate that a dataset has the expected YOLO structure.
    
    Args:
        dataset_path: Path to the dataset directory
        
    Returns:
        bool: True if valid structure, False otherwise
    """
    required_subdirs = ['train', 'test', 'valid']  # or at least one of them
    yaml_files = ['data.yaml', 'dataset.yaml']
    
    # Check for data.yaml or dataset.yaml
    has_yaml = any((dataset_path / yaml_file).exists() for yaml_file in yaml_files)
    if not has_yaml:
        return False
    
    # Check for at least one of the required subdirectories
    has_subdir = any((dataset_path / subdir).exists() for subdir in required_subdirs)
    if not has_subdir:
        return False
    
    return True


def get_base_directory() -> Path:
    """
    Get the base directory where main.py is located.
    
    Returns:
        Path: Base directory path
    """
    try:
        # Try to get the directory of the main module
        if hasattr(sys.modules['__main__'], '__file__'):
            return Path(sys.modules['__main__'].__file__).resolve().parent
        else:
            # Fallback to current file's parent directory (2 levels up from utils/)
            return Path(__file__).resolve().parents[1]
    except Exception:
        # Final fallback to current working directory
        return Path.cwd()


def ensure_directory_exists(directory_path: Path, description: str = "directory") -> bool:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory_path: Path to the directory
        description: Description for logging purposes
        
    Returns:
        bool: True if directory exists or was created successfully, False otherwise
    """
    try:
        directory_path.mkdir(parents=True, exist_ok=True)
        return True
    except OSError as e:
        print(f"❌ Failed to create {description}: {directory_path}")
        print(f"   Error: {e}")
        return False


def list_directory_contents(directory_path: Path, pattern: str = "*") -> List[Path]:
    """
    List contents of a directory matching a pattern.
    
    Args:
        directory_path: Path to the directory
        pattern: Glob pattern to match (default: "*" for all files)
        
    Returns:
        List[Path]: List of matching paths
    """
    try:
        if directory_path.exists() and directory_path.is_dir():
            return list(directory_path.glob(pattern))
        else:
            return []
    except Exception as e:
        print(f"⚠️  Error listing directory contents: {e}")
        return []
