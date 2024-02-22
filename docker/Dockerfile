FROM tensorflow/tensorflow:2.8.0-gpu-jupyter

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get -y update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-dev \
    git \
    vim

COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt

RUN jupyter notebook --generate-config --allow-root
EXPOSE 8888

RUN mkdir -p /workspace
WORKDIR /workspace
