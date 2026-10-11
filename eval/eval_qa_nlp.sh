export PYTHONPATH=$PYTHONPATH:/home/data/nas_hdd/cbx/pvllm
export HF_ENDPOINT=https://hf-mirror.com

export HF_HOME=/home/data/nas_hdd/cbx/hf_cache
export HF_DATASETS_CACHE=/home/data/nas_hdd/cbx/hf_cache
export TMPDIR=/home/data/nas_hdd/cbx/tmp_cache

python eval/eval_qa_nlp.py

# HF_ENDPOINT=https://hf-mirror.com PYTHONPATH=$PYTHONPATH:/home/data/nas_hdd/cbx/pvllm