#!bin/bash

data_dir=/home/caetano/enoe2/
checks_dir=/media/data/checkpoints/
logs_dir=/media/data/logs/

cd ..

docker run --gpus all \
	    -p 8888:8888 \
	    -it \
	    --rm \
	    -v ${PWD}/:/workspace \
	    -v $data_dir:/enoe \
	    -v $checks_dir:/models/checkpoints \
	    -v $logs_dir:/models/logs \
	    tf-flood \
	    bash
