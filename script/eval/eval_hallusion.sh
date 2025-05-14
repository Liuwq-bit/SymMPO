#!/bin/bash

ckpt_path=$1
base_path=${2:-"none"}
openai_key=${3:-"123"}
gpu_id=${4:-"0"}
conv_mode=${5:-"llava_v1"}
data_dir=${6:-"data/hallusion_bench"}

base_name="${ckpt_path}"
answer_file_name=${base_name}
current_time=$(date "+%Y-%m-%d--%H-%M-%S")
log_file=${ckpt_path}/hallusion_bench_${current_time}.log

echo "log_file: "$log_file
echo "ckpt_path: "$ckpt_path
echo "data_dir: "$data_dir
echo "gpu_id: "$gpu_id
echo "conv_mode: "$conv_mode


CUDA_VISIBLE_DEVICES=$gpu_id \
PYTHONPATH=./:$PYTHONPATH \
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
python muffin/eval/inference_hallusion_bench.py \
    --model-path $ckpt_path \
    --question-file ${data_dir}/HallusionBench.json \
    --image-path $data_dir \
    --answers-file ${data_dir}/model_output/${answer_file_name}.json \
    --model-base $base_path \
    --temperature 0.0 \
    --conv-mode $conv_mode \
    --num_beams 3


echo "========>Done generating answers<========"

echo "========>Start evaluating answers<========"


(PYTHONPATH=./:$PYTHONPATH \
python eval/hallusion_evaluation.py \
    --apikey $openai_key \
    --base-path ${data_dir}/model_output/${answer_file_name} \
    --answers-file ${data_dir}/model_output/${answer_file_name}.json \
    --output-path ${data_dir}/model_output/${answer_file_name}_score.jsonl \
&& cat ${data_dir}/model_output/${answer_file_name}_score.jsonl | tee $log_file) &