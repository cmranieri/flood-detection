FROM tensorflow/tensorflow:2.8.0-gpu-jupyter

RUN apt-get -y update && apt-get install -y \
    git \
    vim

RUN pip install opencv-python==4.5.5.62 &\
	pip install opencv-contrib-python==4.5.5.62 &\
	pip install matplotlib==3.4.3 &\
	pip install pandas==1.4.1 &\
	pip install seaborn==0.11.2 &\
	pip install scikit-learn==1.0.2 &\
	pip install scikit-image==0.19.1

RUN mkdir -p /workspace
WORKDIR /workspace
