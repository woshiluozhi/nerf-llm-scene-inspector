#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-nerf-llm-scene-inspector}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "NeRF-LLM Scene Inspector environment setup"
echo
echo "Warning: full Nerfstudio/LERF training requires an NVIDIA GPU, CUDA-compatible PyTorch, and Tiny CUDA NN."
echo "Use Python 3.10 for the broadest Nerfstudio compatibility."
echo

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found on PATH."
  echo "Install Miniconda or Mambaforge, then run:"
  echo "  conda create -n ${ENV_NAME} python=3.10 -y"
  echo "  conda activate ${ENV_NAME}"
else
  if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "Conda environment '${ENV_NAME}' already exists."
  else
    echo "Creating conda environment '${ENV_NAME}'..."
    conda create -n "${ENV_NAME}" python=3.10 -y
  fi
  echo "Installing this project's lightweight Python dependencies..."
  conda run -n "${ENV_NAME}" python -m pip install --upgrade pip
  conda run -n "${ENV_NAME}" python -m pip install -e "${PROJECT_ROOT}[dev,video,dashboard]"
  echo "Activate it with:"
  echo "  conda activate ${ENV_NAME}"
fi

echo
echo "If you prefer to install manually after activating the environment:"
echo "  python -m pip install --upgrade pip"
echo "  python -m pip install -e \".[dev,video,dashboard]\""
echo
echo "Install Nerfstudio:"
echo "  python -m pip install nerfstudio"
echo "  ns-install-cli"
echo "  ns-process-data --help"
echo "  ns-train -h"
echo
echo "Install FFmpeg and COLMAP if missing:"
echo "  conda install -c conda-forge ffmpeg colmap"
echo
echo "Install LERF:"
echo "  git clone https://github.com/kerrj/lerf"
echo "  cd lerf"
echo "  python -m pip install -e ."
echo "  ns-install-cli"
echo "  ns-train -h"
echo
echo "Optional OpenNeRF backend:"
echo "  git clone https://github.com/opennerf/opennerf"
echo "  cd opennerf"
echo "  python -m pip install -e ."
echo "  ns-install-cli"
echo
echo "Expected check: ns-train -h should list lerf, lerf-lite, and lerf-big."
