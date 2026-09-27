#!/usr/bin/env bash
set -e

# -------- OS & paths --------
UNAME="$(uname -s)"

case "$UNAME" in
  Linux*|Darwin*)
    PYTHON=${PYTHON:-python3}
    VENV_BIN=".venv/bin"
    PIP="$VENV_BIN/pip"
    ACTIVATE="$VENV_BIN/activate"
    ;;
  CYGWIN*|MINGW*|MSYS*|WindowsNT)
    # Running in Git Bash or similar on Windows
    # Prefer 'py' launcher if available, else python
    if command -v py >/dev/null 2>&1; then
      PYTHON=${PYTHON:-py}
    else
      PYTHON=${PYTHON:-python}
    fi
    VENV_BIN=".venv/Scripts"
    PIP="$VENV_BIN/pip.exe"
    ACTIVATE="$VENV_BIN/activate"         # for Git Bash
    ACTIVATE_PS1="$VENV_BIN/Activate.ps1" # for PowerShell
    ACTIVATE_BAT="$VENV_BIN/activate.bat" # for CMD
    ;;
  *)
    # default to Unix-style
    PYTHON=${PYTHON:-python3}
    VENV_BIN=".venv/bin"
    PIP="$VENV_BIN/pip"
    ACTIVATE="$VENV_BIN/activate"
    echo "⚠️ Unknown system ($UNAME), assuming Unix-like paths."
    ;;
esac

echo "Using Python: $($PYTHON --version)"

# -------- Create venv --------
if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  $PYTHON -m venv .venv
else
  echo "Virtual environment already exists."
fi

# -------- Install deps --------
if [ -f "requirements.txt" ]; then
  echo "Installing dependencies from requirements.txt..."
  "$PIP" install --upgrade pip
  "$PIP" install -r requirements.txt
else
  echo "No requirements.txt found. Installing example packages..."
  "$PIP" install --upgrade pip
  "$PIP" install torch ultralytics opencv-python
fi

echo "✅ Setup complete!"

# -------- Activation --------
case "$UNAME" in
  Linux*|Darwin*)
    echo "Activating venv for Linux/macOS..."
    # Launch a new interactive bash with venv sourced
    echo "  source ".venv/bin/activate" to activate the venv"
    ;;
  CYGWIN*|MINGW*|MSYS*|WindowsNT)
    echo ""
    echo "👉 Activate the environment on Windows:"
    echo "  • PowerShell:  $ACTIVATE_PS1"
    echo "  • CMD:         $ACTIVATE_BAT"
    echo "  • Git Bash:    source \"$ACTIVATE\""
    ;;
  *)
    echo ""
    echo "👉 Activate manually:"
    echo "  source \"$ACTIVATE\""
    ;;
esac

echo " type deactivate on the terminal to exit the venv"
