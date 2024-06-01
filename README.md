<h2 align="center">
  <img src="resources/logo_black_purple.jpg" width="800">
</h2>

<h3 align="center">
    A Comprehensive Benchmark Suite for Topological Deep Learning
</h3>

<p align="center">
Assess how your model compares against state-of-the-art topological neural networks.
</p>

<div align="center">

[![Lint](https://github.com/pyt-team/TopoBenchmarkX/actions/workflows/lint.yml/badge.svg)](https://github.com/pyt-team/TopoBenchmarkX/actions/workflows/lint.yml)
[![Test](https://github.com/pyt-team/TopoBenchmarkX/actions/workflows/python-app.yml/badge.svg)](https://github.com/pyt-team/TopoBenchmarkX/actions/workflows/python-app.yml)
[![Python](https://img.shields.io/badge/python-3.10+-blue?logo=python)](https://www.python.org/)
[![license](https://badgen.net/github/license/pyt-team/TopoBenchmarkX?color=green)](https://github.com/pyt-team/TopoBenchmarkX/blob/main/LICENSE)
[![slack](https://img.shields.io/badge/chat-on%20slack-purple?logo=slack)](https://join.slack.com/t/pyt-teamworkspace/shared_invite/zt-2k63sv99s-jbFMLtwzUCc8nt3sIRWjEw)


</div>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#toolbox-get-started">Get Started</a> •
  <a href="https://github.com/pyt-team/TopoBenchmarkX/tree/dev/tutorials">Tutorials</a> •
  <a href="#spiral_notepad-references">References</a> 
</p>


<p align="center">
  <img src="resources/photo1716928649.jpeg" width="700">
</p>


## Overview

`TopoBenchmarkX` (TBX) is a Python library developed to train and compare the performances of topological neural networks using different topological domains. Here, a _topological domain_ means a graph, a simplicial complex, a cellular complex, or a hypergraph.

The main pipeline trains and evaluates a wide range of state-of-the-art neural networks (see [:gear: Neural Networks](https://github.com/pyt-team/TopoBenchmarkX/blob/ninamiolane-readme/README.md#gear-neural-networks)) on numerous and varied datasets and benchmark tasks. 

Additionally, the library offers the ability to transform, i.e., _lift_, each dataset from one topological domain to another (see [:top: Liftings](https://github.com/pyt-team/TopoBenchmarkX/blob/ninamiolane-readme/README.md#top-liftings)).

## :toolbox: Get Started

### Create Environment

First, clone the `TopoBenchmarkX` repository and set up a conda environment `tbx` with python 3.11.3.
```
git clone git@github.com:pyt-team/topobenchmarkx.git
cd topobenchmarkx
conda create -n tbx python=3.11.3
```

Next, check the CUDA version of your machine:
```
/usr/local/cuda/bin/nvcc --version
```
and ensure that it matches the CUDA version specified in the `env_setup.sh` file (`CUDA=cu121` by default). If it does not match, update `env_setup.sh` accordingly.

Next, create the environment with the following command.

```
source env_setup.sh
```
This command installs the `TopoBenchmarkX` library and its dependencies. 

### Run Training Pipeline

```
python topobenchmarkx/train.py 
```

### Explore the Tutorials

To add a new dataset and benchmark task, you can explore our [tutorials](https://github.com/pyt-team/TopoBenchmarkX/tree/main/tutorials).

## :gear: Neural Networks

We list the neural networks trained and evaluated by `TopoBenchmarkX`, organized by the topological domain over which they operate: graph, simplicial complex, cellular complex or hypergraph. Many of these neural networks were originally implemented in [`TopoModelX`](https://github.com/pyt-team/TopoModelX).
## Additional details on project
<details>
<summary><b>Hierarchy of configuration files</b></summary>

```
├── configs                   <- Hydra configs
│   ├── callbacks                <- Callbacks configs
│   ├── dataset                  <- Dataset configs
│   │   ├── graph                    <- Graph dataset configs
│   │   ├── hypergraph               <- Hypergraph dataset configs
│   │   └── simplicial               <- Simplicial dataset configs
│   ├── debug                    <- Debugging configs
│   ├── evaluator                <- Evaluator configs
│   ├── experiment               <- Experiment configs
│   ├── extras                   <- Extra utilities configs
│   ├── hparams_search           <- Hyperparameter search configs
│   ├── hydra                    <- Hydra configs
│   ├── local                    <- Local configs
│   ├── logger                   <- Logger configs
│   ├── loss                     <- Loss function configs
│   ├── model                    <- Model configs
│   │   ├── cell                     <- Cell model configs
│   │   ├── graph                    <- Graph model configs
│   │   ├── hypergraph               <- Hypergraph model configs
│   │   └── simplicial               <- Simplicial model configs
│   ├── optimizer                <- Optimizer configs
│   ├── paths                    <- Project paths configs
│   ├── scheduler                <- Scheduler configs
│   ├── trainer                  <- Trainer configs
│   ├── transforms               <- Data transformation configs
│   │   ├── data_manipulations       <- Data manipulation transforms
│   │   ├── dataset_defaults         <- Default dataset transforms
│   │   ├── feature_liftings         <- Feature lifting transforms
│   │   └── liftings                 <- Lifting transforms
│   │       ├── graph2cell               <- Graph to cell lifting transforms
│   │       ├── graph2hypergraph         <- Graph to hypergraph lifting transforms
│   │       ├── graph2simplicial         <- Graph to simplicial lifting transforms
│   │       ├── graph2cell_default.yaml  <- Default graph to cell lifting config
│   │       ├── graph2hypergraph_default.yaml <- Default graph to hypergraph lifting config
│   │       ├── graph2simplicial_default.yaml <- Default graph to simplicial lifting config
│   │       ├── no_lifting.yaml           <- No lifting config
│   │       ├── custom_example.yaml       <- Custom example transform config
│   │       └── no_transform.yaml         <- No transform config
│   ├── wandb_sweep              <- Weights & Biases sweep configs
│   │
│   ├── __init__.py              <- Init file for configs module
│   └── run.yaml               <- Main config for training
```


</details>



### Graphs
| Model | Reference |
| --- | --- |
| GAT | [Graph Attention Networks](https://openreview.net/pdf?id=rJXMpikCZ) |
| GIN | [How Powerful are Graph Neural Networks?](https://openreview.net/pdf?id=ryGs6iA5Km) |
| GCN | [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/pdf/1609.02907v4) |

### Simplicial complexes
| Model | Reference |
| --- | --- |
| SAN | [Simplicial Attention Neural Networks](https://arxiv.org/pdf/2203.07485) |
| SCCN | [Efficient Representation Learning for Higher-Order Data with Simplicial Complexes](https://openreview.net/pdf?id=nGqJY4DODN) |
| SCCNN | [Convolutional Learning on Simplicial Complexes](https://arxiv.org/pdf/2301.11163) |
| SCN | [Simplicial Complex Neural Networks](https://ieeexplore.ieee.org/document/10285604) |

### Cellular complexes
| Model | Reference |
| --- | --- |
| CAN | [Cell Attention Network](https://arxiv.org/pdf/2209.08179) |
| CCCN | [A learning algorithm for computational connected cellular network](https://ieeexplore.ieee.org/document/1202221) |
| CCXN | [Cell Complex Neural Networks](https://openreview.net/pdf?id=6Tq18ySFpGU) |
| CWN | [Weisfeiler and Lehman Go Cellular: CW Networks](https://arxiv.org/pdf/2106.12575) |

### Hypergraphs
| Model | Reference |
| --- | --- |
| AllDeepSet | [You are AllSet: A Multiset Function Framework for Hypergraph Neural Networks](https://openreview.net/pdf?id=hpBTIv2uy_E) |
| AllSetTransformer | [You are AllSet: A Multiset Function Framework for Hypergraph Neural Networks](https://openreview.net/pdf?id=hpBTIv2uy_E) |
| EDGNN | [Equivariant Hypergraph Diffusion Neural Operators](https://arxiv.org/pdf/2207.06680) |
| UniGNN | [UniGNN: a Unified Framework for Graph and Hypergraph Neural Networks](https://arxiv.org/pdf/2105.00956) |
| UniGNN2 | [UniGNN: a Unified Framework for Graph and Hypergraph Neural Networks](https://arxiv.org/pdf/2105.00956) |

## :top: Liftings

We list the liftings used in `TopoBenchmarkX` to transform datasets. Here, a _lifting_ refers to a function that transforms a dataset defined on a topological domain (_e.g._, on a graph) into the same dataset but supported on a different topological domain (_e.g._, on a simplicial complex).

### Graph2Simplicial
| Name | Description | Reference |
| --- | --- | --- |
| CliqueLifting | The algorithm finds the cliques in the graph and creates simplices. Given a clique the first simplex added is the one containing all the nodes of the clique, then the simplices composed of all the possible combinations with one node missing, then two nodes missing, and so on, until all the possible pairs are added. Then the method moves to the next clique. | [Simplicial Complexes](https://en.wikipedia.org/wiki/Clique_complex) |
| KHopLifting | For each node in the graph, the algorithm finds the set of nodes that are at most k connections away from the initial node. This set is then treated as if it was a clique from the CliqueLifting method. The process is repeated for all nodes in the graph. | [Neighborhood Complexes](https://arxiv.org/pdf/math/0512077) |

### Graph2Cell
| Name | Description | Reference |
| --- | --- | --- |
| CycleLifting | The algorithm finds a cycle base for the graph. Given this set of cycles the method creates a cell for each one. | [CW Complexes](https://en.wikipedia.org/wiki/CW_complex) |

### Graph2Hypergraph
| Name | Description | Reference |
| --- | --- | --- |
| KHopLifting | For each node in the graph, the algorithm finds the set of nodes that are at most k connections away from the initial node. This set is then used to create an hyperedge. The process is repeated for all nodes in the graph. | [Section 3.4](https://ieeexplore.ieee.org/abstract/document/9264674) |
| KNearestNeighborsLifting | For each node in the graph, the method finds the k nearest nodes by using the Euclidean distance between the vectors of features. The set of k nodes found is considered as an hyperedge. The proces is repeated for all nodes in the graph. | [Section 3.1](https://ieeexplore.ieee.org/abstract/document/9264674) |

## :keyboard: Development

To join the development of `TopoBenchmarkX`, you should install the library in dev mode. 

For this, you can create an environment using either conda or docker. Both options are detailed below.

### Using conda env

If you don't have conda on your machine, please follow [their guide](https://docs.anaconda.com/free/miniconda/miniconda-install/) to install it. 

We recommend using Python 3.11.3, which is the python version used to run the unit-tests. You can create create and activate a conda environment as follows:
   ```bash
   conda create -n tbx python=3.11.3
   conda activate tbx
   ```

Then:

1. Clone a copy of tbx from source:

   ```bash
   git clone git@github.com:pyt-team/topobenchmarkx.git
   cd topobenchmarkx
   ```

2. Install the required dependencies:

   ```bash
   bash env_setup.sh
   ```
   **Notes:**
   - Modify `install_requirements.sh` to select the proper `CUDA` and `torch` versions among the available options (`CUDA=cu121` and `torch=2.3.0` by default).
   - Please check [this website](https://github.com/pyg-team/pyg-lib) to check the combination that works best for you.

5. Ensure that you have a working tbx installation by running the entire test suite with

   ```bash
   pytest
   ```

    In case an error occurs, please first check if all sub-packages ([`torch-scatter`](https://github.com/rusty1s/pytorch_scatter), [`torch-sparse`](https://github.com/rusty1s/pytorch_sparse), [`torch-cluster`](https://github.com/rusty1s/pytorch_cluster) and [`torch-spline-conv`](https://github.com/rusty1s/pytorch_spline_conv)) are on its latest reported version.

6. Install pre-commit hooks:

   ```bash
   pre-commit install
   ```



### Using Docker

For ease of use, TopoBenchmarkX employs <img src="https://github.com/wesbos/Font-Awesome-Docker-Icon/blob/master/docker-white.svg" width="20" height="20"> [Docker](https://www.docker.com/). To set it up on your system you can follow [their guide](https://docs.docker.com/get-docker/). once installed, please follow the next steps:

First, navigate to the correct folder.
```
cd /path/to/TopoBenchmarkX
```

Then we need to build the Docker image.
```
docker build -t topobenchmark:new .
```

Depending if you want to use GPUs or not, these are the commands to run the Docker image and mount the current directory.

With GPUs
```
docker run -it -d --gpus all --volume $(pwd):/TopoBenchmarkX topobenchmark:new
```

With CPU
```
docker run -it -d --volume $(pwd):/TopoBenchmarkX topobenchmark:new
```

Happy development!

## :spiral_notepad: References

To learn more about `TopoBenchmarkX`, we invite you to read the paper:

```
@misc{topobenchmarkx2024,
      title={TopoBenchmarkX},
      author={PyT-Team},
      year={2024},
      eprint={TBD},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
If you find `TopoBenchmarkX` useful, we appreciate if you cite it!
