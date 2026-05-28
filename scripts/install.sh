#!/usr/bin/env bash
set -euo pipefail

python -m pip install -r requirements.txt
python -m pip install submodules/depth-diff-gaussian-rasterization
python -m pip install submodules/simple-knn
