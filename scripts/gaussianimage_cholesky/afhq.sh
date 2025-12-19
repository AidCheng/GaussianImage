#!/bin/bash

data_path=$1
num_points=2048

if [ -z "$data_path" ]; then
    echo "Error: No data_path provided."
    echo "Usage: $0 <data_path>"
    exit 1
fi

CUDA_VISIBLE_DEVICES=0 python train_quantize.py -d $data_path \
--data_name afhq --model_name GaussianImage_Cholesky --num_points $num_points --iterations 50000 \
--model_path ./checkpoints/afhq/GaussianImage_Cholesky_50000_$num_points

CUDA_VISIBLE_DEVICES=0 python test_quantize.py -d $data_path \
--data_name afhq --model_name GaussianImage_Cholesky --num_points $num_points --iterations 50000 \
--model_path ./checkpoints_quant/afhq/GaussianImage_Cholesky_50000_$num_points

CUDA_VISIBLE_DEVICES=0 python fit_gaussian.py -d $data_path \
--data_name afhq --model_name GaussianImage_Cholesky --num_points $num_points
--model_path ./checkpoints_quant/afhq/GaussianImage_Cholesky_50000_$num_points