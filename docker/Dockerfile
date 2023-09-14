FROM nvidia/cuda:11.6.0-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update

RUN apt install -y g++ \
         make \
         wget \
         unzip \
         python3-pip

RUN ln -s /usr/bin/python3 /usr/bin/python
# Set mirror site if you are in China
# RUN python3 -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# install PyTorch
RUN python3 -m pip install torch==1.12.0+cu116 -f https://download.pytorch.org/whl/torch_stable.html

# install PyG
RUN pip3 install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv torch_geometric -f https://data.pyg.org/whl/torch-1.12.0+cu116.html

RUN mkdir /working_dir
WORKDIR /working_dir