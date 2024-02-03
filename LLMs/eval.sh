#!/bin/bash


python eval.py \
    --model_path="/data/checkpoints/llama7b/"\
    --dataset_dir=/data/datasets/Food/Recipe1M/ \
    --save_dir=./results/ \
    --eval_type="im_t_ing" \
    --model_type="LLAMA" \
    --partition="test_1kimage" \
    --prompt_type="ing_i_t__instruct"\
    --batch_size 1



'''

python eval.py \
    --model_path="/data/checkpoints/llama7b/"\
    --dataset_dir=/data/datasets/Food/Recipe1M/ \
    --save_dir=./results/ \
    --eval_type="im_t_ing" \
    --model_type="LLAMA" \
    --partition="test_1kimage" \
    --prompt_type="ing_i_t__instruct"\
    --batch_size 1


python eval.py \
    --model_path="/data/checkpoints/Mistral/Mistral-T5-7B-v1/"\
    --dataset_dir=/data/datasets/Food/Recipe1M/ \
    --save_dir=./results/ \
    --eval_type="im_t_ing" \
    --model_type="Mistral" \
    --partition="test_1kimage" \
    --prompt_type="ing_i_t__instruct"\
    --batch_size 1


python eval.py \
    --model_path="microsoft/phi-2"\
    --dataset_dir=/data/datasets/Food/Recipe1M/ \
    --save_dir=./results/ \
    --eval_type="im_t_ing" \
    --model_type="Phi2" \
    --partition="test_1kimage" \
    --prompt_type="ing_i_t__instruct"\
    --batch_size 1


python eval.py \
    --model_path="gpt2" \
    --dataset_dir=/data/datasets/Food/Recipe1M/ \
    --save_dir=./results/ \
    --eval_type="im_t_ing" \
    --model_type="GTP2" \
    --partition="test_1kimage" \
    --prompt_type="ing_i_t__instruct"\
    --batch_size 1


# Salesforce/instructblip-flan-t5-xl
# "Salesforce/instructblip-vicuna-7b"

python eval.py \
    --model_path=Salesforce/instructblip-flan-t5-xl \
    --dataset_dir=/data/datasets/Food/Recipe1M/ \
    --save_dir=./results/ \
    --eval_type="im_t_ing" \
    --model_type="InstructBlip" \
    --partition="test_1kimage" \
    --prompt_type="ing_i_t__instruct"\
    --batch_size 1

'''
