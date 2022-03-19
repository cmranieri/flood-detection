#!/usr/bin/bash

python /workspace/src/main.py --split 1 --config_path ../configs/gray_flow_config.yaml 
python /workspace/src/main.py --split 2 --config_path ../configs/gray_flow_config.yaml 
python /workspace/src/main.py --split 3 --config_path ../configs/gray_flow_config.yaml 
python /workspace/src/main.py --split 4 --config_path ../configs/gray_flow_config.yaml 
python /workspace/src/main.py --split 4 --config_path ../configs/enoe_config.yaml 

