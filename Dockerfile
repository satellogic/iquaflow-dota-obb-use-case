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

RUN conda create -n iqf python=3.6 -q -y && \
	conda run -n iqf pip install git+https://gitlab+deploy-token-45:FKSA3HpmgUoxa5RZ69Cf@git.satellogic.team/iqf/iquaflow-@add-testds

CMD ["/bin/bash", "-c", "./start.sh && /bin/bash"]