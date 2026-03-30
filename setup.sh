#!/usr/bin/env bash
# ============================================================
#  setup.sh  –  Swarm Intelligence Exploration
#  Sets up a Python virtual environment and installs deps.
#  Supports macOS and Linux (bash/zsh).
# ============================================================

set -euo pipefail

VENV_DIR=".venv"
REPO_NAME="Swarm Intelligence Exploration"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Colour

info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }

echo ""
echo "  ==========================================="
echo "   ${REPO_NAME} – Setup"
echo "  ==========================================="
echo ""

# --- Locate Python -----------------------------------------------
PYTHON=""
for candidate in python3 python3.12 python3.11 python3.10 python3.9 python; do
    if command -v "$candidate" &>/dev/null; then
        ver=$("$candidate" --version 2>&1 | awk '{print $2}')
        major=$(echo "$ver" | cut -d. -f1)
        minor=$(echo "$ver" | cut -d. -f2)
        if [ "$major" -ge 3 ] && [ "$minor" -ge 9 ]; then
            PYTHON="$candidate"
            info "Found $PYTHON ($ver)"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    error "Python 3.9+ not found. Install it via:"
    echo "  macOS:   brew install python  OR  https://www.python.org/downloads/"
    echo "  Ubuntu:  sudo apt install python3 python3-venv python3-pip"
    exit 1
fi

# macOS: warn if Pygame requires SDL2 (user may need brew)
if [[ "$(uname)" == "Darwin" ]]; then
    warn "macOS detected. If Pygame fails to install, run:"
    echo "       brew install sdl2 sdl2_image sdl2_mixer sdl2_ttf"
fi

# --- Create virtual environment ----------------------------------
if [ -d "$VENV_DIR" ]; then
    info "Virtual environment already exists at ${VENV_DIR}/ — skipping creation."
else
    info "Creating virtual environment in ${VENV_DIR}/ ..."
    "$PYTHON" -m venv "$VENV_DIR"
    info "Virtual environment created."
fi

# --- Upgrade pip -------------------------------------------------
info "Upgrading pip ..."
"$VENV_DIR/bin/python" -m pip install --upgrade pip --quiet

# --- Install dependencies ----------------------------------------
info "Installing dependencies from requirements.txt ..."
"$VENV_DIR/bin/pip" install -r requirements.txt

echo ""
echo "  ==========================================="
echo "   Setup complete!"
echo "  ==========================================="
echo ""
echo "  Activate the environment:"
echo "    source ${VENV_DIR}/bin/activate"
echo ""
echo "  Then run a simulation, e.g.:"
echo "    python boids-sim.py"
echo "    python aco-maze.py"
echo "    python boids-maze.py"
echo "    python swarm-maze-fast.py"
echo ""
