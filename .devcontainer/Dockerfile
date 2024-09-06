ARG UBUNTU_VERSION=22.04
ARG NVIDIA_CUDA_VERSION=11.8.0

#
# Docker builder stage.
#
FROM nvidia/cuda:${NVIDIA_CUDA_VERSION}-cudnn8-devel-ubuntu${UBUNTU_VERSION} AS build

ARG COLMAP_GIT_COMMIT=main
ARG CUDA_ARCHITECTURES=all
ENV QT_XCB_GL_INTEGRATION=xcb_egl

# Prevent stop building ubuntu at time zone selection.
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends --no-install-suggests \
        build-essential g++ wget unzip pkg-config software-properties-common libssl-dev

WORKDIR /root
RUN wget https://github.com/Kitware/CMake/releases/download/v3.25.2/cmake-3.25.2.tar.gz
RUN tar -xzvf cmake-3.25.2.tar.gz
RUN cd cmake-3.25.2 && ./bootstrap && make -j$(nproc) && make install
RUN rm cmake-3.25.2.tar.gz

# Prepare and empty machine for building.
RUN apt-get update && \
    apt-get install -y --no-install-recommends --no-install-suggests \
        git \
        ninja-build \
        build-essential
        
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH

RUN conda create -yn citygs python=3.9 pip
RUN conda run -n citygs --no-capture-output pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
RUN conda init bash