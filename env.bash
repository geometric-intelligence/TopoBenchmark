# #!/bin/bash

conda create -n topox15 python=3.11.3
conda activate topox15

pip install --upgrade pip
pip install -e '.[all]'

pip install --no-dependencies git+https://github.com/pyt-team/TopoNetX.git
pip install --no-dependencies git+https://github.com/pyt-team/TopoModelX.git

CUDA="cu115" # if available, select the CUDA version suitable for your system
             # e.g. cpu, cu102, cu111, cu113, cu115
pip install torch==2.0.1 --extra-index-url https://download.pytorch.org/whl/${CUDA}
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.1+${CUDA}.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+${CUDA}.html
#pip install torch-geometric -f https://data.pyg.org/whl/torch-2.6.0.dev20240506+${CUDA}.html

# pytest

# pre-commit install
