#!/usr/bin/env bash
set -euo pipefail

# Expanded cleanup: removes common artifacts produced during development and tests

echo "==> Cleaning dataset outputs"
rm -rf ./unified || true
rm -rf ./unified_reshuffled || true
rm -rf ./unified_from_shuffled || true

echo "==> Cleaning test and run artifacts"
rm -rf ./runs || true
rm -rf ./reports/htmlcov || true
rm -rf ./.pytest_cache || true
rm -rf ./__pycache__ || true
find . -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true

# Clean user_system test fixture outputs if present
rm -rf ./tests/compile_out* || true
rm -rf ./tests/compile_no_dupes || true
rm -rf ./tests/compile_names || true
rm -rf ./tests/compile_out_shuffled* || true
rm -rf ./tests/test_shuffle_output || true

# Optional: coverage data files
rm -f ./.coverage* || true

echo "Cleanup completed."
