# #!/bin/bash

yes | conda create -n topox python=3.11.3
conda activate topox

pip install --upgrade pip
pip install -e '.[all]'

yes | pip install git+https://github.com/pyt-team/TopoNetX.git
yes | pip install git+https://github.com/pyt-team/TopoModelX.git
yes | pip install git+https://github.com/pyt-team/TopoEmbedX.git

CUDA="cu115" # if available, select the CUDA version suitable for your system
             # e.g. cpu, cu102, cu111, cu113, cu115
yes | pip install torch==2.0.1 --extra-index-url https://download.pytorch.org/whl/${CUDA}
yes | pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.1+${CUDA}.html
yes | pip install torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+${CUDA}.html

pytest

pre-commit install
