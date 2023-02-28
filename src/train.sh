#!/usr/bin/bash

python /workspace/src/main.py --eval_only true --split 1 --config_path ../configs/flow_config.yaml 
python /workspace/src/main.py --eval_only true --split 2 --config_path ../configs/flow_config.yaml 
python /workspace/src/main.py --eval_only true --split 3 --config_path ../configs/flow_config.yaml 
python /workspace/src/main.py --eval_only true --split 4 --config_path ../configs/flow_config.yaml 

python /workspace/src/main.py --eval_only true --split 1 --config_path ../configs/single_rgb_config.yaml 
python /workspace/src/main.py --eval_only true --split 2 --config_path ../configs/single_rgb_config.yaml 
python /workspace/src/main.py --eval_only true --split 3 --config_path ../configs/single_rgb_config.yaml 
python /workspace/src/main.py --eval_only true --split 4 --config_path ../configs/single_rgb_config.yaml 

python /workspace/src/main.py --eval_only true --split 1 --config_path ../configs/single_gray_flow_config.yaml 
python /workspace/src/main.py --eval_only true --split 2 --config_path ../configs/single_gray_flow_config.yaml 
python /workspace/src/main.py --eval_only true --split 3 --config_path ../configs/single_gray_flow_config.yaml 
python /workspace/src/main.py --eval_only true --split 4 --config_path ../configs/single_gray_flow_config.yaml 

python /workspace/src/main.py --eval_only true --split 1 --config_path ../configs/stack_flow_config.yaml 
python /workspace/src/main.py --eval_only true --split 2 --config_path ../configs/stack_flow_config.yaml 
python /workspace/src/main.py --eval_only true --split 3 --config_path ../configs/stack_flow_config.yaml 
python /workspace/src/main.py --eval_only true --split 4 --config_path ../configs/stack_flow_config.yaml 

python /workspace/src/main.py --eval_only true --split 1 --config_path ../configs/single_flow_diffs_config.yaml 
python /workspace/src/main.py --eval_only true --split 2 --config_path ../configs/single_flow_diffs_config.yaml 
python /workspace/src/main.py --eval_only true --split 3 --config_path ../configs/single_flow_diffs_config.yaml 
python /workspace/src/main.py --eval_only true --split 4 --config_path ../configs/single_flow_diffs_config.yaml 

python /workspace/src/main.py --eval_only true --split 1 --config_path ../configs/single_gray_flow_diffs_config.yaml 
python /workspace/src/main.py --eval_only true --split 2 --config_path ../configs/single_gray_flow_diffs_config.yaml 
python /workspace/src/main.py --eval_only true --split 3 --config_path ../configs/single_gray_flow_diffs_config.yaml 
python /workspace/src/main.py --eval_only true --split 4 --config_path ../configs/single_gray_flow_diffs_config.yaml 

python /workspace/src/main.py --eval_only true --split 1 --config_path ../configs/stack_flow_diffs_config.yaml 
python /workspace/src/main.py --eval_only true --split 2 --config_path ../configs/stack_flow_diffs_config.yaml 
python /workspace/src/main.py --eval_only true --split 3 --config_path ../configs/stack_flow_diffs_config.yaml 
python /workspace/src/main.py --eval_only true --split 4 --config_path ../configs/stack_flow_diffs_config.yaml 

python /workspace/src/main.py --eval_only true --split 1 --config_path ../configs/pair_gray_diffs_config.yaml 
python /workspace/src/main.py --eval_only true --split 2 --config_path ../configs/pair_gray_diffs_config.yaml 
python /workspace/src/main.py --eval_only true --split 3 --config_path ../configs/pair_gray_diffs_config.yaml 
python /workspace/src/main.py --eval_only true --split 4 --config_path ../configs/pair_gray_diffs_config.yaml 

