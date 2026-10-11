export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_MODE=disabled
export PYTHONPATH=$PYTHONPATH:/home/data/nas_hdd/cbx/pvllm

export HF_HOME=/home/data/nas_hdd/cbx/hf_cache
export HF_DATASETS_CACHE=/home/data/nas_hdd/cbx/hf_cache
export TMPDIR=/home/data/nas_hdd/cbx/tmp_cache

accelerate launch \
    --mixed_precision 'bf16' \
    trl/scripts/grpo_pllm_dr.py \
    --model_name_or_path /home/data/nas_hdd/cbx/pvllm/models/base_model \
    --dataset_name /home/data/nas_hdd/cbx/pvllm/sft_data_v4/train \
    --output_dir /home/data/nas_hdd/cbx/pvllm/output_grpo_dr_geo_125 \
    --learning_rate 1e-5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_generations 8 \
    --max_steps 2000 \
    --max_completion_len 2048 \
    --grpo_gamma 0.01 \
    --logging_steps 8 \
    --save_steps 50 \
    --warmup_steps 10 \
    --reward_func pvl_dr_reward \
    --report_to wandb \
    
