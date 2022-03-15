#!bin/bash

data_dir=/home/caetano/enoe2/
models_dir=/media/data/models/
logs_dir=/media/data/logs/

cd ..

docker run --gpus all \
	    -p 8888:8888 \
	    -it \
	    --rm \
	    -v ${PWD}/:/workspace \
	    -v $data_dir:/enoe \
	    -v $models_dir:/models \
	    tf-flood \
	    bash
