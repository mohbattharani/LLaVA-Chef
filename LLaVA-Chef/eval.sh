#!/bin/bash


python eval.py \
    --model_path="/data/checkpoints/LLaVA/LLaVA-Chef/LLaVA-Chef-S3/"\
    --dataset_dir=/data/datasets/Food/Recipe1M/ \
    --save_dir=./results/ \
    --eval_type="im_t" \
    --model_type="LLaVAChefS3" \
    --partition="test_1kimage" \
    --prompt_type="i__ing" \
    --start=0 \
    --end=-1
'''

python eval.py \
    --model_path="/data/checkpoints/LLaVA/LLaVA-7B-Lightening-v1-1/"\
    --dataset_dir=/data/datasets/Food/Recipe1M/ \
    --save_dir=./results/ \
    --eval_type="im_t_ing" \
    --model_type="LLaVA" \
    --partition="test_1kimage" \
    --prompt_type="ing_i_t__instruct" \
    --start=0 \
    --end=-1


python eval.py \
    --model_path="/data/checkpoints/LLaVA/LLaVA-Chef/LLaVA-Chef-S1/"\
    --dataset_dir=/data/datasets/Food/Recipe1M/ \
    --save_dir=./results/ \
    --eval_type="im_t_ing" \
    --model_type="LLaVAChefS1" \
    --partition="test_1kimage" \
    --prompt_type="ing_i_t__instruct" \
    --start=0 \
    --end=-1


python eval.py \
    --model_path="/data/checkpoints/LLaVA/LLaVA-Chef/LLaVA-Chef-S2/"\
    --dataset_dir=/data/datasets/Food/Recipe1M/ \
    --save_dir=./results/ \
    --eval_type="im_t_ing" \
    --model_type="LLaVAChefS2" \
    --partition="test_1kimage" \
    --prompt_type="ing_i_t__instruct" \
    --start=0 \
    --end=-1


python eval.py \
    --model_path="/data/checkpoints/LLaVA/LLaVA-Chef/LLaVA-Chef-S3/"\
    --dataset_dir=/data/datasets/Food/Recipe1M/ \
    --save_dir=./results/ \
    --eval_type="im_t_ing" \
    --model_type="LLaVAChefS3" \
    --partition="test_1kimage" \
    --prompt_type="ing_i_t__instruct" \
    --start=0 \
    --end=-1

'''