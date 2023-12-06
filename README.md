# TopoBenchmarkX

# Initialize the Container

## Manual Docker Commands

```bash
# Navigate to the TopoModelX directory
cd /path/to/TopoModelX

# Build the Docker image
docker build -t topobenchmark:new .

# Run the Docker image interactively with GPUs and mount the current directory into the container
docker run -it -d --gpus all --volume $(pwd):/TopoBenchmarkX topobenchmark:new

# Run the Docker image interactively with CPU and mount the current directory into the container
docker run -it -d --volume $(pwd):/TopoBenchmarkX topobenchmark:new
