#!/bin/bash

model_path="./checkpoints/llava-7b-finetune/"
dataset_dir="/data/mohbat/datasets/Recipe1M/"
prompt_type="ing_i_t__instruct"
save_dir="./llava/eval/results/"

python llava/eval/eval_recipe2.py \
    --model_path=$model_path \
    --dataset_dir=$dataset_dir \
    --save_dir=$save_dir \
    --prompt_type=$prompt_type \
    --eval_type="im_t_ing" \
    --model_type="llavachefv3" \
    --start=0 \
    --partition test_image 

'''
python llava/eval/eval_recipe2.py \
    --model_path=$model_path \
    --dataset_dir=$dataset_dir \
    --save_dir=$save_dir \
    --prompt_type=$prompt_type \
    --eval_type="im" \
    --model_type="llavachef" \
    --start=0 \
    --end=5000 \
    --partition test_1kimage 


python llava/eval/eval_recipe2.py \
    --model_path=$model_path \
    --dataset_dir=$dataset_dir \
    --save_dir=$save_dir \
    --prompt_type=$prompt_type \
    --eval_type="t" \
    --model_type="llavachef" \
    --start=0 \
    --end=5000 \
    --partition test_1kimage 
   
python llava/eval/eval_recipe2.py \
    --model_path=$model_path \
    --dataset_dir=$dataset_dir \
    --save_dir=$save_dir \
    --prompt_type=$prompt_type \
    --eval_type="im_t" \
    --model_type="llavachef" \
    --start=0 \
    --end=5000 \
    --partition test_1kimage 
    

python llava/eval/eval_recipe2.py \
    --model_path=$model_path \
    --dataset_dir=$dataset_dir \
    --save_dir=$save_dir \
    --prompt_type=$prompt_type \
    --eval_type="im_ing" \
    --model_type="llavachef" \
    --start=0 \
    --end=5000 \
    --partition test_1kimage 


python llava/eval/eval_recipe2.py \
    --model_path=$model_path \
    --dataset_dir=$dataset_dir \
    --save_dir=$save_dir \
    --prompt_type=$prompt_type \
    --eval_type="t_ing" \
    --model_type="llavachef" \
    --start=0 \
    --end=5000 \
    --partition test_1kimage

'''
