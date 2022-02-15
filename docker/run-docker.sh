#!bin/bash

data_dir=/media/caetano/Caetano/enoe/

cd ..

docker run --gpus all \
	   -it \
	   --rm \
	   -v ${PWD}/:/workspace \
	   -v $data_dir:/enoe \
	   tf-flood \
	   bash
