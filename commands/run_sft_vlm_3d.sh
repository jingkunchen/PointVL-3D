export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_MODE=disabled

export HF_HOME=/home/data/nas_hdd/cbx/hf_cache
export HF_DATASETS_CACHE=/home/data/nas_hdd/cbx/hf_cache
export TMPDIR=/home/data/nas_hdd/cbx/tmp_cache

accelerate launch \
    --config_file /home/data/nas_hdd/cbx/pvllm/configs/zero2.yaml \
    /home/data/nas_hdd/cbx/pvllm/trl/scripts/sft_pvlm_3d.py \
    --dataset_name /home/data/nas_hdd/cbx/pvllm/sft_data_v4/train \
    --model_name_or_path /home/data/nas_hdd/cbx/pvllm/models/Qwen2.5-VL-3B-Instruct \
    --gradient_accumulation_steps 2 \
    --output_dir /home/data/nas_hdd/cbx/pvllm/output_dir_3d \
    --dtype bfloat16 \
    --per_device_train_batch_size 4 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --max_steps 3000 \
