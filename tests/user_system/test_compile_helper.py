"""
Test suite for user_system.utils.compile_helper module.

This module tests the compile helper functionality including
name parsing, JSON/YAML handling, and argument processing.
"""

import sys
import tempfile
import json
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import pytest

# Put repo root on sys.path
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Import the module under test
from user_system.utils.compile_helper import (
    parse_names_block,
    parse_names_arg
)


class TestParseNamesBlock:
    """Test cases for parse_names_block function."""

    def test_parse_names_block_none(self):
        """Test parsing None input."""
        result = parse_names_block(None)
        assert result == {}

    def test_parse_names_block_empty_dict(self):
        """Test parsing empty dictionary."""
        result = parse_names_block({})
        assert result == {}

    def test_parse_names_block_list(self):
        """Test parsing list input."""
        names_list = ["cow", "sheep", "chicken"]
        result = parse_names_block(names_list)
        expected = {0: "cow", 1: "sheep", 2: "chicken"}
        assert result == expected

    def test_parse_names_block_dict_string_keys(self):
        """Test parsing dictionary with string keys."""
        names_dict = {"0": "cow", "1": "sheep", "2": "chicken"}
        result = parse_names_block(names_dict)
        expected = {0: "cow", 1: "sheep", 2: "chicken"}
        assert result == expected

    def test_parse_names_block_dict_int_keys(self):
        """Test parsing dictionary with integer keys."""
        names_dict = {0: "cow", 1: "sheep", 2: "chicken"}
        result = parse_names_block(names_dict)
        expected = {0: "cow", 1: "sheep", 2: "chicken"}
        assert result == expected

    def test_parse_names_block_dict_mixed_keys(self):
        """Test parsing dictionary with mixed key types."""
        names_dict = {"0": "cow", 1: "sheep", "2": "chicken"}
        result = parse_names_block(names_dict)
        expected = {0: "cow", 1: "sheep", 2: "chicken"}
        assert result == expected

    def test_parse_names_block_dict_invalid_keys(self):
        """Test parsing dictionary with invalid keys."""
        names_dict = {"invalid": "cow", "also_invalid": "sheep"}
        result = parse_names_block(names_dict)
        assert result == {}

    def test_parse_names_block_dict_partial_invalid_keys(self):
        """Test parsing dictionary with some invalid keys."""
        names_dict = {"0": "cow", "invalid": "sheep", "1": "chicken"}
        result = parse_names_block(names_dict)
        expected = {0: "cow", 1: "chicken"}
        assert result == expected

    def test_parse_names_block_empty_list(self):
        """Test parsing empty list."""
        result = parse_names_block([])
        assert result == {}

    def test_parse_names_block_list_with_none_values(self):
        """Test parsing list with None values."""
        names_list = ["cow", None, "chicken"]
        result = parse_names_block(names_list)
        expected = {0: "cow", 1: "None", 2: "chicken"}
        assert result == expected

    def test_parse_names_block_list_with_non_string_values(self):
        """Test parsing list with non-string values."""
        names_list = ["cow", 123, "chicken"]
        result = parse_names_block(names_list)
        expected = {0: "cow", 1: "123", 2: "chicken"}
        assert result == expected


class TestParseNamesArg:
    """Test cases for parse_names_arg function."""

    def test_parse_names_arg_none(self):
        """Test parsing None argument."""
        result = parse_names_arg(None)
        assert result is None

    def test_parse_names_arg_empty_string(self):
        """Test parsing empty string."""
        result = parse_names_arg("")
        assert result is None

    def test_parse_names_arg_inline_json(self):
        """Test parsing inline JSON string."""
        json_str = '{"0": "cow", "1": "sheep"}'
        result = parse_names_arg(json_str)
        expected = {0: "cow", 1: "sheep"}
        assert result == expected

    def test_parse_names_arg_inline_json_with_names_key(self):
        """Test parsing inline JSON with names key."""
        json_str = '{"names": {"0": "cow", "1": "sheep"}}'
        result = parse_names_arg(json_str)
        # The function doesn't handle "names" key in inline JSON, only in files
        # It returns an empty dict because parse_names_block gets {"names": {...}} but doesn't find valid int keys
        assert result == {}

    def test_parse_names_arg_inline_json_list(self):
        """Test parsing inline JSON list."""
        json_str = '["cow", "sheep", "chicken"]'
        result = parse_names_arg(json_str)
        expected = {0: "cow", 1: "sheep", 2: "chicken"}
        assert result == expected

    def test_parse_names_arg_invalid_json(self):
        """Test parsing invalid JSON string."""
        invalid_json = '{"0": "cow", "1": "sheep"'  # Missing closing brace
        result = parse_names_arg(invalid_json)
        assert result is None

    def test_parse_names_arg_yaml_file(self):
        """Test parsing YAML file."""
        yaml_content = """
names:
  0: cow
  1: sheep
  2: chicken
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            yaml_path = f.name
        
        try:
            result = parse_names_arg(yaml_path)
            expected = {0: "cow", 1: "sheep", 2: "chicken"}
            assert result == expected
        finally:
            Path(yaml_path).unlink(missing_ok=True)

    def test_parse_names_arg_yml_file(self):
        """Test parsing .yml file."""
        yaml_content = """
names:
  0: cow
  1: sheep
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write(yaml_content)
            yml_path = f.name
        
        try:
            result = parse_names_arg(yml_path)
            expected = {0: "cow", 1: "sheep"}
            assert result == expected
        finally:
            Path(yml_path).unlink(missing_ok=True)

    def test_parse_names_arg_json_file(self):
        """Test parsing JSON file."""
        json_content = {
            "names": {
                "0": "cow",
                "1": "sheep",
                "2": "chicken"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(json_content, f)
            json_path = f.name
        
        try:
            result = parse_names_arg(json_path)
            expected = {0: "cow", 1: "sheep", 2: "chicken"}
            assert result == expected
        finally:
            Path(json_path).unlink(missing_ok=True)

    def test_parse_names_arg_json_file_direct_list(self):
        """Test parsing JSON file with direct list."""
        json_content = ["cow", "sheep", "chicken"]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(json_content, f)
            json_path = f.name
        
        try:
            result = parse_names_arg(json_path)
            expected = {0: "cow", 1: "sheep", 2: "chicken"}
            assert result == expected
        finally:
            Path(json_path).unlink(missing_ok=True)

    def test_parse_names_arg_nonexistent_file(self):
        """Test parsing nonexistent file."""
        result = parse_names_arg("./nonexistent.yaml")
        assert result is None

    def test_parse_names_arg_non_yaml_json_file(self):
        """Test parsing file with non-YAML/JSON extension."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("cow\nsheep\nchicken")
            txt_path = f.name
        
        try:
            result = parse_names_arg(txt_path)
            # Should try to parse as inline JSON
            assert result is None
        finally:
            Path(txt_path).unlink(missing_ok=True)

    def test_parse_names_arg_yaml_file_read_error(self):
        """Test parsing YAML file with read error."""
        yaml_path = "./test.yaml"
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.read_text', side_effect=OSError("Read error")):
            with pytest.raises(OSError):
                parse_names_arg(yaml_path)

    def test_parse_names_arg_yaml_parse_error(self):
        """Test parsing YAML file with parse error."""
        yaml_content = "invalid: yaml: content: ["
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            yaml_path = f.name
        
        try:
            with pytest.raises(Exception):  # yaml.scanner.ScannerError
                parse_names_arg(yaml_path)
        finally:
            Path(yaml_path).unlink(missing_ok=True)

    def test_parse_names_arg_json_file_read_error(self):
        """Test parsing JSON file with read error."""
        json_path = "./test.json"
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.read_text', side_effect=OSError("Read error")):
            with pytest.raises(OSError):
                parse_names_arg(json_path)

    def test_parse_names_arg_json_parse_error(self):
        """Test parsing JSON file with parse error."""
        json_content = '{"0": "cow", "1": "sheep"'  # Missing closing brace
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write(json_content)
            json_path = f.name
        
        try:
            with pytest.raises(Exception):  # json.decoder.JSONDecodeError
                parse_names_arg(json_path)
        finally:
            Path(json_path).unlink(missing_ok=True)

    def test_parse_names_arg_yaml_file_no_names_key(self):
        """Test parsing YAML file without names key."""
        yaml_content = """
other_data:
  some_value: test
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            yaml_path = f.name
        
        try:
            result = parse_names_arg(yaml_path)
            assert result == {}
        finally:
            Path(yaml_path).unlink(missing_ok=True)

    def test_parse_names_arg_json_file_no_names_key(self):
        """Test parsing JSON file without names key."""
        json_content = {
            "other_data": {
                "some_value": "test"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(json_content, f)
            json_path = f.name
        
        try:
            result = parse_names_arg(json_path)
            assert result == {}
        finally:
            Path(json_path).unlink(missing_ok=True)

    def test_parse_names_arg_yaml_file_non_dict_root(self):
        """Test parsing YAML file with non-dict root."""
        yaml_content = """
- cow
- sheep
- chicken
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            yaml_path = f.name
        
        try:
            result = parse_names_arg(yaml_path)
            expected = {0: "cow", 1: "sheep", 2: "chicken"}
            assert result == expected
        finally:
            Path(yaml_path).unlink(missing_ok=True)

    def test_parse_names_arg_json_file_non_dict_root(self):
        """Test parsing JSON file with non-dict root."""
        json_content = ["cow", "sheep", "chicken"]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(json_content, f)
            json_path = f.name
        
        try:
            result = parse_names_arg(json_path)
            expected = {0: "cow", 1: "sheep", 2: "chicken"}
            assert result == expected
        finally:
            Path(json_path).unlink(missing_ok=True)


class TestCompileHelperIntegration:
    """Integration tests for compile helper functions."""

    def test_parse_names_workflow(self):
        """Test complete names parsing workflow."""
        # Test inline JSON
        json_str = '{"0": "cow", "1": "sheep"}'
        result1 = parse_names_arg(json_str)
        assert result1 == {0: "cow", 1: "sheep"}
        
        # Test list parsing
        result2 = parse_names_block(["cow", "sheep", "chicken"])
        assert result2 == {0: "cow", 1: "sheep", 2: "chicken"}
        
        # Test dict parsing
        result3 = parse_names_block({"0": "cow", "1": "sheep"})
        assert result3 == {0: "cow", 1: "sheep"}

    def test_file_parsing_workflow(self):
        """Test file parsing workflow."""
        # Create test YAML file
        yaml_content = """
names:
  0: cow
  1: sheep
  2: chicken
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            yaml_path = f.name
        
        try:
            # Parse the file
            result = parse_names_arg(yaml_path)
            expected = {0: "cow", 1: "sheep", 2: "chicken"}
            assert result == expected
            
            # Test that the parsed result works with parse_names_block
            block_result = parse_names_block(result)
            assert block_result == expected
            
        finally:
            Path(yaml_path).unlink(missing_ok=True)

    def test_error_handling_workflow(self):
        """Test error handling workflow."""
        # Test invalid JSON
        invalid_json = '{"0": "cow", "1": "sheep"'  # Missing closing brace
        result1 = parse_names_arg(invalid_json)
        assert result1 is None
        
        # Test invalid file
        result2 = parse_names_arg("./nonexistent.yaml")
        assert result2 is None
        
        # Test invalid keys in dict
        result3 = parse_names_block({"invalid": "cow", "also_invalid": "sheep"})
        assert result3 == {}
        
        # Test None inputs
        result4 = parse_names_arg(None)
        assert result4 is None
        
        result5 = parse_names_block(None)
        assert result5 == {}

    def test_edge_cases_workflow(self):
        """Test edge cases workflow."""
        # Test empty inputs
        result1 = parse_names_arg("")
        assert result1 is None
        
        result2 = parse_names_block([])
        assert result2 == {}
        
        result3 = parse_names_block({})
        assert result3 == {}
        
        # Test mixed data types
        result4 = parse_names_block(["cow", 123, None, "chicken"])
        expected = {0: "cow", 1: "123", 2: "None", 3: "chicken"}
        assert result4 == expected
        
        # Test partial invalid keys
        result5 = parse_names_block({"0": "cow", "invalid": "sheep", "1": "chicken"})
        expected = {0: "cow", 1: "chicken"}
        assert result5 == expected
