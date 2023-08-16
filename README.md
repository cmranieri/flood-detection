# flood-detection

## Data
The dataset used in this research is available at:
{https://drive.google.com/drive/folders/1FaUFubP3UlUKQ9nGAOkdYZ_ynmmC5dej?usp=drive_link}

## Docker

Using Docker images is a way of improving reproducibility of a project, as well as automating install, setup, and running of the code.
Hence, we provided a Dockerfile and supporting scripts for building and running the container.
By following these instructions, you will have a GPU-enabled container running, with all the dependencies for this project.

If you don't have Docker installed yet:
1. Install Docker to your machine. Follow the instructions [here](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository). We recommend performing the post-install steps, so that you don't need to run the containers as root (i.e., sudo).
2. Install the NVIDIA support for Docker. Follow the instructions [here](https://github.com/NVIDIA/nvidia-docker).

After you have Docker installed with NVIDIA support, proceed with the following.

To build the Docker container, run:
```
cd docker
bash build-docker.sh
```

To run a Docker container, change the variables in ```run-docker.sh``` to those in your local machine.
For example, if you stored the data in the ```/home/data``` diretory, set ```$data_dir='/home/data'```.
The other variables, ```$checks_dir``` and ```$logs_dir```, refer to the checkpoints and logs generated while training the deep learning models.
Please, set them as your convenience.
After setting up the paths, run:
```
bash run-docker.sh
```

To launch jupyter notebook from within the Docker container, run:
```
jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root
```
