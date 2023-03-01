#!bin/bash

data_dir=/home/caetano/enoe2/
flow_dir=/home/caetano/enoe_flow/
rgbdiffs_dir=/home/caetano/rgbdiffs
models_dir=/media/data/models/

cd ..

docker run --gpus all \
	    -p 8888:8888 \
        -p 6006:6006 \
	    -it \
	    --rm \
        -u $(id -u):$(id -g) \
	    -v ${PWD}/:/workspace \
	    -v $data_dir:/enoe \
	    -v $flow_dir:/flow \
	    -v $rgbdiffs_dir:/rgbdiffs \
	    -v $models_dir:/models \
	    tf-flood \
	    bash
