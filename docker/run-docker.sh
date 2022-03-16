#!bin/bash

#data_dir=/home/caetano/enoe2/
data_dir=/media/caetano/Caetano/enoe2/
flow_dir=/home/caetano/flow/
models_dir=/media/data/models/

cd ..

docker run --gpus all \
	    -p 8888:8888 \
	    -it \
	    --rm \
        -u $(id -u):$(id -g) \
	    -v ${PWD}/:/workspace \
	    -v $data_dir:/enoe \
	    -v $flow_dir:/flow \
	    -v $models_dir:/models \
	    tf-flood \
	    bash
