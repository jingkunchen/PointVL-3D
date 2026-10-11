# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# /// script
# dependencies = [
#     "trl",
#     "Pillow>=9.4.0",
#     "peft",
#     "trackio",
#     "kernels",
# ]
# ///

import os

import torch
from datasets import load_dataset
from transformers import AutoModelForImageTextToText

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer_PLLM_3d,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)


# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")

if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.max_length = None

    ################
    # Model, Tokenizer & Processor
    ################
    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        # Passing None would not be treated the same as omitting the argument, so we include it only when valid.
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config

    from model_arch.modeling_qwen2_5_pvl_batch_3d import Qwen2_5_PVLForConditionalGeneration
    model = Qwen2_5_PVLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path, 
        trust_remote_code=model_args.trust_remote_code, 
        ignore_mismatched_sizes=True,
        **model_kwargs
    )
    print(f"load point bert ckpt...")
    ckpt_pb = torch.load('/home/data/nas_hdd/cbx/pvllm/model_arch/point_bert/Point-BERT.pth', weights_only=False)
    ckpt_pb = {k.replace('module.', ''): v for k, v in ckpt_pb['base_model'].items()}
    model.model.pc_extractor.point_bert.load_state_dict(ckpt_pb, assign=True)
    print(f"load point bert ckpt done.")
    ################
    # Dataset
    ################
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    ################
    # Training
    ################
    trainer = SFTTrainer_PLLM_3d(
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
    )

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
