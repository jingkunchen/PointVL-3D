export CUDA_VISIBLE_DEVICES=4,5,6,7
export WANDB_MODE=disabled

export HF_HOME=/home/data/nas_hdd/cbx/hf_cache
export HF_DATASETS_CACHE=/home/data/nas_hdd/cbx/hf_cache
export TMPDIR=/home/data/nas_hdd/cbx/tmp_cache

export CUDA_VISIBLE_DEVICES=4,5,6,7
export WANDB_MODE=disabled
accelerate launch \
    --config_file /home/data/nas_hdd/cbx/pvllm/configs/zero2.yaml \
    /home/data/nas_hdd/cbx/pvllm/trl/scripts/sft_vlm.py \
    --dataset_name /home/data/nas_hdd/cbx/pvllm/sft_data_v4_2d/train \
    --model_name_or_path /home/data/nas_hdd/cbx/pvllm/models/llama3-llava-next-8b-hf \
    --gradient_accumulation_steps 8 \
    --output_dir /home/data/nas_hdd/cbx/pvllm/output_dir_llava \
    --dtype bfloat16 \
    --per_device_train_batch_size 1 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --max_steps 3000 \

    # --config_file /home/data/nas_hdd/cbx/pvllm/configs/zero3.yaml \