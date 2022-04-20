#!/usr/bin/bash

python /workspace/src/main.py --split 2 --config_path ../configs/stack_flow_diffs_config.yaml 
python /workspace/src/main.py --split 2 --config_path ../configs/stack_flow_config.yaml 
python /workspace/src/main.py --split 2 --config_path ../configs/flow_diffs_config.yaml 
python /workspace/src/main.py --split 2 --config_path ../configs/flow_config.yaml 
python /workspace/src/main.py --split 2 --config_path ../configs/enoe_config.yaml 
python /workspace/src/main.py --split 2 --config_path ../configs/gray_flow_config.yaml 

