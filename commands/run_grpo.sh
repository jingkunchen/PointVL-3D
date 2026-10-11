export CUDA_VISIBLE_DEVICES=4,5,6,7
export WANDB_MODE=disabled

accelerate launch \
    --mixed_precision 'bf16' \
    trl/scripts/grpo_pllm.py \
    --model_name_or_path /home/data/nas_hdd/cbx/pvllm/out_dir_v3 \
    --dataset_name /data/ruoshi/pcdata/sft_data_v4/train \
    --output_dir /home/data/nas_hdd/cbx/pvllm/out_dir_grpo_vcs \
    --learning_rate 5e-6 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_generations 8 \
    --max_steps 2000 \
    --max_completion_len 2048 \
    --grpo_gamma 0.01 \
    --logging_steps 8 \
    --save_steps 50 \
    --warmup_steps 10 \
    --reward_func pvl_reward \
    --report_to wandb \
    
