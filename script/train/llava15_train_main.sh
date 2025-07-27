#!/bin/bash

echo "----------Start llava15_train----------"

task_name=llava15_7b_DPO
exp_name=${1:-"logps"}
ckpt=${2:-"liuhaotian/llava-v1.5-7b"}
vision_tower=${3:-"openai/clip-vit-large-patch14-336"}
raw_data_path=${4:-"demo_data/similar"}
gpu_vis=${5:-"0,1,2,3"}
learning_rate=${6:-5e-6}
lambda=${7:-0.5}
data_dir=${raw_data_path}-with-logps

echo "task_name: "$task_name
echo "exp_name: "$exp_name
echo "raw_data_path: "$raw_data_path
echo "gpu_vis: "$gpu_vis
echo "learning_rate: "$learning_rate
echo "lamda: "$lamda

MASTER_PORT_START=10000
MASTER_PORT_END=65535
MASTER_PORT="$(
	comm -23 \
		<(seq "${MASTER_PORT_START}" "${MASTER_PORT_END}" | sort) \
		<(ss -Htan | awk '{ print $4 }' | awk -F ':' '{ print $NF }' | sort -u) |
		shuf | head -n 1
)"


PYTHONPATH=./:$PYTHONPATH \
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
deepspeed --include localhost:$gpu_vis  --master_port $MASTER_PORT \
    --module muffin.train.train_llava15 \
    --deepspeed script/zero3_offload.json \
    --version llava_v1 \
    --model_name_or_path $ckpt \
    --raw_data_path $raw_data_path \
    --data_dir $data_dir \
    --image_folder not_used \
    --vision_tower $vision_tower \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --save_only_model True \
    --fully_tune True \
    --image_aspect_ratio pad \
    --bf16 True \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --output_dir result/symmpo/$task_name-$exp_name \
    --num_train_epochs 10 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --data_source_names '' \
    --data_source_weights 1 \
    --max_steps 670 \
    --learning_rate $learning_rate \
    --weight_decay 0.01 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 2 \
    --logging_dir checkpoint/$task_name-$exp_name/log \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --task DPO \
    --report_to tensorboard \
    --run_name $exp_name \
    --dataloader_num_workers 16 \
    --dpo_use_average False \
    --dpo_token_weighted False \
    --dpo_token_weight 1.0 \
    --dpo_beta_1 0.1 \
    --dpo_beta_2 0.1 \
    --lamda $lamda \
    --save_steps 335 \
    --save_total_limit 4 \