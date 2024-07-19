# GNNDrive

GNNDrive is a disk-based GNN training framework, specifically designed to optimize the training on large-scale graphs using a single machine with ordinary hardware components, such as CPU, GPU, and limited memory. GNNDrive minimizes the memory footprint for feature extraction to reduce the memory contention between sampling and extracting. It also introduces asynchronous feature extraction to mitigate I/O congestion for massive data movements. 

## Installation and Running

Follow the instructions below to install the requirements and run an example using [ogbn_papers100M](https://ogb.stanford.edu/docs/nodeprop/#ogbn-papers100M) dataset.

### Installation

1. Clone our library
    ```shell
    git clone 
    ```

2. Run Docker
    1. Build docker images
        ```shell
        cd docker
        docker build -t GNN:gpu .
        ```

    2. Install nvidia-container-runtime for docker
        ```shell
        curl -s -L https://nvidia.github.io/nvidia-container-runtime/gpgkey | \
        sudo apt-key add -
        distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
        curl -s -L https://nvidia.github.io/nvidia-container-runtime/$distribution/nvidia-container-runtime.list | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-runtime.list
        sudo apt-get update

        sudo apt-get install nvidia-container-runtime
        ```
    
        > Please refer the following links for more details.
        > 1. https://docs.docker.com/config/containers/resource_constraints/#gpu
        > 2. https://nvidia.github.io/nvidia-container-runtime/

    3. Run container with limited memory
        ```shell
        docker run --gpus all -it --ipc=host \
            --name GNN-16g --memory 16G --memory-swap 32G \
            -v /path-to-file:/working_dir/ GNN:gpu bash
        ```
    
        > Note: `--memory` limits the maximum amount of memory the container can use.
        >
        > Please refer the following link for more details.
        > 1. https://docs.docker.com/config/containers/resource_constraints/#limit-a-containers-access-to-memory


3. Install necessary library. 
    1. liburing
        ```shell
        # download
        wget https://github.com/axboe/liburing/archive/refs/tags/liburing-2.1.zip
        unzip liburing-2.1.zip

        # install
        cd liburing-2.1
        ./configure --cc=gcc --cxx=g++;
        make -j$(nproc);
        make install;
        ```

        > Please refer the following link for more details.
        > 1. https://github.com/axboe/liburing

    2. Ninja
        ```shell
        wget https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip
        unzip ninja-linux.zip -d /usr/local/bin/
        update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force
        ```

### Running

1. Prepare dataset
    ```shell
    python3 prepare_dataset_ogbn.py
    ```

2. Preprocess for baseline, i.e., Ginex
    ```shell
    python3 create_neigh_cache.py --neigh-cache-size 6000000000
    ````

5. Run baselines
    ```shell
    # run PyG+
    python3 run_baseline.py

    # run Ginex
    python3 run_ginex.py --neigh-cache-size 6000000000 \
        --feature-cache-size 6000000000 --sb-size 1500
    ```

6. Run GNNDrive
    ```shell
    # run without data parallelism in GPU 
    python3 run_async.py --compute-type gpu

    # run without data parallelism in CPU 
    python3 run_async.py --compute-type cpu

    # run with data parallelism using 2 subprocesses in GPU 
    python3 run_async_multi.py --compute-type gpu \
        --world-size 2

    # run with data parallelism using 2 subprocesses in CPU 
    python3 run_async_multi.py --compute-type cpu \
        --world-size 2
    ```

    > Note: 
    > 1. `--compute-type` indicates that the system uses GPU or CPU when training.
    > 2. `--world-size` indicates the number of subprocesses used for training.



## Maintainer

Qisheng Jiang (jiangqsh@shanghaitech.edu.cn)

Lei Jia (jialei2022@shanghaitech.edu.cn)

## Acknowledgements

We thank authors of [Ginex](https://dl.acm.org/doi/10.14778/3551793.3551819) for providing the source code of Ginex and PyG+. Our implementation uses some funtions  of [Ginex](https://github.com/SNU-ARC/Ginex.git).
