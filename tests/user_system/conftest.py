import shutil
from pathlib import Path
import pytest


def _clean_local_tmp_dirs(base_dir: Path) -> None:
	"""Remove temporary test artifacts under tests/user_system.

	This targets ad-hoc directories like tmp_inference_* and tmp_* that some
	tests may create under the test package directory (rather than using
	pytest's tmp_path). The cleanup is best-effort and ignores errors.
	"""
	for pattern in ("tmp_inference_*", "tmp_*"):
		for p in base_dir.glob(pattern):
			try:
				shutil.rmtree(p, ignore_errors=True)
			except Exception:
				# Best-effort; ignore cleanup failures
				pass


@pytest.fixture(autouse=True)
def _autoclean_tmp_dirs():
	"""Autouse fixture that cleans stray tmp directories before and after tests.

	Ensures test runs do not leave residual folders like tests/user_system/tmp_*.
	"""
	base_dir = Path(__file__).parent
	# Pre-test cleanup (in case previous run left artifacts)
	_clean_local_tmp_dirs(base_dir)
	yield
	# Post-test cleanup
	_clean_local_tmp_dirs(base_dir)


