"""
utils/modify_helper.py

Dataset label modification functionality for the YOLO livestock detection system.
This module provides functions to modify class labels in YOLO datasets.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import shutil
import yaml
import json


def load_dataset_yaml(dataset_path: Path) -> Dict[str, Any]:
    """Load dataset configuration from YAML file."""
    yaml_files = ["dataset.yaml", "data.yaml"]
    
    for yaml_file in yaml_files:
        yaml_path = dataset_path / yaml_file
        if yaml_path.exists():
            try:
                with open(yaml_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                print(f"Warning: Could not load {yaml_path}: {e}")
                continue
    
    raise FileNotFoundError(f"No valid dataset YAML found in {dataset_path}")


def save_dataset_yaml(dataset_path: Path, data: Dict[str, Any]) -> None:
    """Save dataset configuration to YAML file."""
    yaml_path = dataset_path / "dataset.yaml"
    
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


def find_label_files(dataset_path: Path) -> List[Path]:
    """Find all label files in the dataset."""
    label_files = []
    
    # Look for labels in standard YOLO structure
    for split in ["train", "val", "valid", "test"]:
        labels_dir = dataset_path / "labels" / split
        if labels_dir.exists():
            label_files.extend(labels_dir.glob("*.txt"))
        
        # Also check for split/labels structure
        labels_dir = dataset_path / split / "labels"
        if labels_dir.exists():
            label_files.extend(labels_dir.glob("*.txt"))
    
    return label_files


def update_label_file(label_file: Path, old_label: str, new_label: str, 
                     label_mapping: Dict[str, str]) -> int:
    """
    Update labels in a single label file.
    Returns the number of labels changed.
    """
    labels_changed = 0
    
    try:
        with open(label_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        updated_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                updated_lines.append(line)
                continue
            
            parts = line.split()
            if len(parts) < 5:  # YOLO format: class_id x y w h
                updated_lines.append(line)
                continue
            
            class_id = parts[0]
            
            # Check if this class_id corresponds to the old_label
            if class_id in label_mapping:
                # Update the class_id to the new one
                parts[0] = label_mapping[class_id]
                labels_changed += 1
            
            updated_lines.append(' '.join(parts))
        
        # Write back the updated content
        with open(label_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(updated_lines))
            
    except Exception as e:
        print(f"Warning: Could not update {label_file}: {e}")
    
    return labels_changed


def modify_dataset_labels(dataset_path: Path, old_label: str, new_label: str, 
                         output_path: Path) -> Dict[str, int]:
    """
    Modify labels in a YOLO dataset.
    
    Args:
        dataset_path: Path to the input dataset
        old_label: Current label name to change
        new_label: New label name
        output_path: Path where modified dataset should be saved
    
    Returns:
        Dictionary with modification statistics
    """
    results = {
        "files_processed": 0,
        "labels_changed": 0,
        "classes_updated": 0
    }
    
    # Load dataset configuration
    try:
        dataset_config = load_dataset_yaml(dataset_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Could not find dataset configuration: {e}")
    
    # Get current class names
    names = dataset_config.get("names", [])
    if isinstance(names, dict):
        # Convert dict to list if needed
        names = [names.get(str(i), f"class_{i}") for i in range(len(names))]
    elif not isinstance(names, list):
        names = []
    
    # Check if old_label exists
    old_label_lower = old_label.lower().strip()
    old_label_index = None
    
    for i, name in enumerate(names):
        if name.lower().strip() == old_label_lower:
            old_label_index = i
            break
    
    if old_label_index is None:
        print(f"Warning: Label '{old_label}' not found in dataset classes: {names}")
        return results
    
    # Create label mapping (old_index -> new_index)
    # If new_label already exists, use its index; otherwise, add it
    new_label_lower = new_label.lower().strip()
    new_label_index = None
    
    for i, name in enumerate(names):
        if name.lower().strip() == new_label_lower:
            new_label_index = i
            break
    
    if new_label_index is None:
        # Add new label
        names.append(new_label)
        new_label_index = len(names) - 1
        results["classes_updated"] = 1
    
    # Create mapping from old index to new index
    label_mapping = {str(old_label_index): str(new_label_index)}
    
    # Copy dataset to output location if different
    if output_path != dataset_path:
        print(f"📁 Copying dataset from {dataset_path} to {output_path}")
        if output_path.exists():
            shutil.rmtree(output_path)
        shutil.copytree(dataset_path, output_path)
        dataset_path = output_path
    
    # Find and update all label files
    label_files = find_label_files(dataset_path)
    print(f"📝 Found {len(label_files)} label files to process")
    
    for label_file in label_files:
        labels_changed = update_label_file(label_file, old_label, new_label, label_mapping)
        results["labels_changed"] += labels_changed
        results["files_processed"] += 1
    
    # Update dataset configuration
    dataset_config["names"] = names
    save_dataset_yaml(dataset_path, dataset_config)
    
    print(f"✅ Updated dataset configuration:")
    print(f"   Old label '{old_label}' (index {old_label_index}) → '{new_label}' (index {new_label_index})")
    print(f"   Total classes: {len(names)}")
    
    return results


def validate_dataset_structure(dataset_path: Path) -> bool:
    """Validate that the dataset has proper YOLO structure."""
    if not dataset_path.exists():
        return False
    
    # Check for dataset configuration
    try:
        load_dataset_yaml(dataset_path)
    except FileNotFoundError:
        return False
    
    # Check for at least one label file
    label_files = find_label_files(dataset_path)
    return len(label_files) > 0
