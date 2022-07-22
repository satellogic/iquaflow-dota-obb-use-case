ARG PYTORCH="1.9.0"
ARG CUDA="11.1"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV CUDALONG=111
ENV TORCH_CUDA_ARCH_LIST="8.6"
#"6.0 6.1 7.0+PTX"

ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

ENV DEBIAN_FRONTEND noninteractive
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
ENV PATH=/usr/local/nvidia/bin:$PATH
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN apt-key del 7fa2af80 && \
	rm -f /etc/apt/sources.list.d/cuda.list && \
	rm -f /etc/apt/sources.list.d/nvidia-ml.list && \
	apt-get update -y && apt-get install wget -y && \
	wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/$(uname -m)/cuda-keyring_1.0-1_all.deb && \
	dpkg -i cuda-keyring_1.0-1_all.deb && \
	apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub && \
	apt update -y

RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 && \
	apt-get update && \
	apt-get install ffmpeg libsm6 libxext6 -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install mmdetection
RUN conda clean --all
RUN mkdir /work && mkdir /work/mmdetection

ENV FORCE_CUDA="1"
WORKDIR /work

COPY . /work/

RUN git clone https://github.com/open-mmlab/mmdetection.git /work/OBBDetection/mmdetection
RUN pip install cython --no-cache-dir
RUN pip install "git+https://github.com/open-mmlab/cocoapi.git#subdirectory=pycocotools"
RUN cd /work/OBBDetection/mmdetection && pip install --no-cache-dir -e .

RUN cd /work/OBBDetection/BboxToolkit && pip install -v -e . && \
	cd /work/OBBDetection && \
	pip install mmcv-full==1.3.9 -f https://download.openmmlab.com/mmcv/dist/cu${CUDALONG}/torch${PYTORCH}/index.html

RUN pip install ipykernel jupyterlab rasterio mlflow

RUN conda create -n iqf python=3.6 -q -y
RUN conda run -n iqf pip install git+https://github.com/satellogic/iquaflow.git
# RUN cd iquaflow && conda run -n iqf pip install . && cd ..

CMD ["/bin/bash", "-c", "./start.sh && /bin/bash"]
