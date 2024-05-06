# #!/bin/bash

# set -e

# # Step 1: Upgrade pip
# pip install --upgrade pip

# # Step 2: Install dependencies
# yes | pip install -e '.[all]'
# yes | pip install --no-dependencies git+https://github.com/pyt-team/TopoNetX.git
# yes | pip install --no-dependencies git+https://github.com/pyt-team/TopoModelX.git
# yes | pip install torch==2.0.1 --extra-index-url https://download.pytorch.org/whl/cu115
# yes | pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.1+cu115.html
# yes | pip install torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cu115.html
# yes | pip install lightning>=2.0.0
# yes | pip install numpy pre-commit jupyterlab notebook ipykernel


yes | conda create -n topox python=3.11.3
conda activate topox

pip install -e '.[all]'

yes | pip install --no-dependencies git+https://github.com/pyt-team/TopoNetX.git
yes | pip install --no-dependencies git+https://github.com/pyt-team/TopoModelX.git

yes | pip install torch==2.0.1 --extra-index-url https://download.pytorch.org/whl/cu115
yes | pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.1+cu115.html
yes | pip install torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cu115.html
yes | pip install numpy pre-commit jupyterlab notebook ipykernel

pytest

pre-commit install
