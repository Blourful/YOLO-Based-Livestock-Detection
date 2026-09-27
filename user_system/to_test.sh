#!/usr/bin/env bash
set -euo pipefail

# Resolve to script directory so paths work regardless of where it's run from
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Ensure output directory exists
mkdir -p reports/htmlcov

# Run tests with coverage (utils and others), produce HTML report under reports/htmlcov
pytest tests ../tests --cov=. --cov-branch \
  --cov-report=term-missing \
  --cov-report=html:reports/htmlcov

# Build absolute path to report; convert to Windows path on Git Bash if available
REPORT_POSIX="${SCRIPT_DIR}/reports/htmlcov/index.html"
if command -v cygpath >/dev/null 2>&1; then
  REPORT_WIN="$(cygpath -w "$REPORT_POSIX")"
else
  REPORT_WIN="$REPORT_POSIX"
fi

# Open the HTML report using an absolute path
python - <<PY
import os, webbrowser, pathlib
report = r"$REPORT_WIN"
if not os.path.exists(report):
    raise SystemExit(f"Report not found: {report}")
try:
    p = pathlib.Path(report)
    url = p.as_uri() if p.exists() else report
except Exception:
    url = report
print(f"Opening {url}")
webbrowser.open(url)
PY
