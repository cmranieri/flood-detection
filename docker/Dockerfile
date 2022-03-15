FROM tensorflow/tensorflow:2.8.0-gpu-jupyter

RUN apt-get -y update && apt-get install -y \
    git \
    vim

COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt

RUN mkdir -p /workspace
WORKDIR /workspace
