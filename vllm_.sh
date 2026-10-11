export HF_HOME=/home/data/nas_hdd/cbx/hf_cache
export HF_DATASETS_CACHE=/home/data/nas_hdd/cbx/hf_cache
export TMPDIR=/home/data/nas_hdd/cbx/tmp_cache
export CUDA_VISIBLE_DEVICES="2,3" 

vllm serve /home/data/nas_hdd/cbx/pvllm/xyaif_ckpt/output_mimo_2d \
    --trust-remote-code \
    --served-model-name qwen_sft \
    --gpu-memory-utilization 0.90 \
    --tensor-parallel-size 2 \
    --port 8000 \
    --max_model_len 5120 \
    --max-num-seqs 32 \
    

