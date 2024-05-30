FROM python:3.11.3

WORKDIR /TopoBenchmarkX

COPY . .

RUN pip install --upgrade pip

RUN pip install -e '.[all]'

# Note that not all combinations of torch and CUDA are available
# See https://github.com/pyg-team/pyg-lib to check the configuration that works for you
RUN TORCH="2.3.0"   
                # available options: 1.12.0, 1.13.0, 2.0.0, 2.1.0, 2.2.0, or 2.3.0
RUN CUDA="cu121"    
                # if available, select the CUDA version suitable for your system
                # available options: cpu, cu102, cu113, cu116, cu117, cu118, or cu121
RUN pip install torch==${TORCH} --extra-index-url https://download.pytorch.org/whl/${CUDA}
RUN pip install pyg-lib torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html