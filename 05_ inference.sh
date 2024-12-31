#!/bin/bash

# 获取脚本所在目录的绝对路径
script_dir=$(dirname $(readlink -f $0))

echo $script_dir

accelerate -m axolotl.cli.inference $script_dir/config/qlora.yml \
    --lora_model_dir="./outputs/qlora-out"