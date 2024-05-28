# :dart: TopoBenchmarkX

Immediately assess how your model stacks against the state-of-the-art of topological models.

<p align="center">
  <img src="resources/photo1716928649.jpeg" width="700">
</p>

## Topological Deep Learning

`TopoBenchmarkX` is a Python library developed to train and compare models using different topological structures.

It offers ready-to-use training pipelines to train and test a model and immediately compare its results with a wide range of models. It also supports numerous and varied datasets, and the ability to transform each dataset from one domain to another.

## :toolbox: Tutorials


## :gear: Models

Many of the models implemented are taken from [`TopoModelX`](https://github.com/pyt-team/TopoModelX).

### Graphs
| Model | Description | Reference |
| --- | --- | --- |
| GAT | - | - |
| GIN | - | - |
| GCN | - | - |

### Simplicial complexes
| Model | Description | Reference |
| --- | --- | --- |
| SAN | - | - |
| SCCN | - | - |
| SCCNN | - | - |
| SCN | - | - |

### Cellular complexes
| Model | Description | Reference |
| --- | --- | --- |
| CAN | - | - |
| CCCN | - | - |
| CCXN | - | - |
| CWN | - | - |

### Hypergraphs
| Model | Description | Reference |
| --- | --- | --- |
| AllDeepSet | - | - |
| AllSetTransformer | - | - |
| EDGNN | - | - |
| UniGNN | - | - |
| UniGNN2 | - | - |

## :top: Liftings
### Graph2Simplicial
| Name | Description | Reference |
| --- | --- | --- |
| CliqueLifting | - | - |
| KHopLifting | - | - |

### Graph2Cell
| Name | Description | Reference |
| --- | --- | --- |
| CycleLifting | - | - |

### Graph2Hypergraph
| Name | Description | Reference |
| --- | --- | --- |
| KHopLifting | - | - |
| KNearestNeighborsLifting | - | - |

## :keyboard: Development

For ease of use, TopoBenchmarkX employs [Docker](https://www.docker.com/). To set it up on your system you can follow [their guide](https://docs.docker.com/get-docker/).

### <img src="https://github.com/wesbos/Font-Awesome-Docker-Icon/blob/master/docker-white.svg" width="30" height="30"> Using Docker

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

## :spiral_notepad: References
