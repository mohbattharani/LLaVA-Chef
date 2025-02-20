#!/bin/bash
    
# --num_gpus=4
# /data/mohbat/models/LLaVA/LLaVA-7B-Lightening-v1-1/
# ./checkpoints/llava-7b-finetune/
deepspeed --num_gpus=2 llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path ./checkpoints/llava-7b-finetune/ \
    --version llava_llama_2 \
    --data_path /data/mohbat/datasets/Recipe1M/ \
    --image_folder /data/mohbat/datasets/Recipe1M/images/ \
    --vision_tower openai/clip-vit-large-patch14 \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --tune_mm_mlp_adapter False \
    --freeze_backbone True \
    --bf16 True \
    --output_dir ./checkpoints/llava-7b-finetune \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to none 
