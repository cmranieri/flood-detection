#!/usr/bin/bash

time python /workspace/src/main.py --split 2 --config_path ../configs/ae_enoe_config.yaml 
time python /workspace/src/main.py --split 3 --config_path ../configs/ae_enoe_config.yaml 
time python /workspace/src/main.py --split 4 --config_path ../configs/ae_enoe_config.yaml 

