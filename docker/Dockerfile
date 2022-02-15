FROM tensorflow/tensorflow:2.7.0-gpu-jupyter

RUN apt-get -y update && apt-get install -y \
    git \
    gcc \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-dev \
    vim

RUN pip install opencv-python \
		opencv-contrib-python \
		matplotlib \
		scipy \
		pandas \
		seaborn \
		sklearn \
		pillow

RUN mkdir -p /workspace
WORKDIR /workspace
