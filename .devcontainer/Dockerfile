FROM python:3.11.3

WORKDIR /TopoBenchmarkX

COPY . .

RUN pip install --upgrade pip

RUN pip install -e '.[all]'
RUN pip install --no-dependencies git+https://github.com/pyt-team/TopoNetX.git
RUN pip install --no-dependencies git+https://github.com/pyt-team/TopoModelX.git@240-review-nncell-and-nncombinatorial-models
RUN pip install torch==2.0.1 --extra-index-url https://download.pytorch.org/whl/cu115
RUN pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.1+cu115.html
RUN pip install torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cu115.html
RUN pip install lightning>=2.0.0
RUN pip install numpy pre-commit jupyterlab notebook ipykernel