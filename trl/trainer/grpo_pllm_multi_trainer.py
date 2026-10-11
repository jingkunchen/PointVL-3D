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

import inspect
import os
import re
import textwrap
from collections import defaultdict, deque
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional, Union

import datasets
import torch
import torch.utils.data
import transformers
from accelerate import logging
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from datasets import Dataset, IterableDataset
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader, Sampler
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.trainer_utils import seed_worker
from transformers.utils import is_datasets_available, is_flash_attn_2_available, is_peft_available, is_rich_available

from ..data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template, prepare_multimodal_messages
from ..extras.profiling import profiling_context, profiling_decorator
from ..extras.vllm_client import VLLMClient
from ..import_utils import is_liger_kernel_available, is_vllm_available
from ..models import prepare_deepspeed, prepare_fsdp, prepare_peft_model, unwrap_model_for_generation
from ..models.utils import _ForwardRedirection
from .callbacks import SyncRefModelCallback
from .grpo_config import GRPOConfig
from .utils import (
    RepeatSampler,
    disable_dropout_in_model,
    ensure_master_addr_port,
    entropy_from_logits,
    generate_model_card,
    get_comet_experiment_url,
    identity,
    nanmax,
    nanmin,
    nanstd,
    pad,
    print_prompt_completions_sample,
    selective_log_softmax,
    shuffle_sequence_dict,
    split_pixel_values_by_grid,
    split_tensor_dict,
    truncate_with_protected_tokens,
    unsplit_pixel_values_by_grid,
)


if is_peft_available():
    from peft import PeftConfig, PeftModel

if is_liger_kernel_available():
    from liger_kernel.chunked_loss import LigerFusedLinearGRPOLoss

if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

if is_wandb_available():
    import wandb


logger = logging.get_logger(__name__)

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class GRPOTrainer_pllm_multi(Trainer):
    _tag_names = ["trl", "grpo"]

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[Union[PreTrainedTokenizerBase, ProcessorMixin]] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        
        model_id = model.config._name_or_path
        if args.model_init_kwargs is not None:
            logger.warning(
                "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                "The `model_init_kwargs` will be ignored."
            )

        # Some models (SmolVLM/Idefics3) don't support `logits_to_keep` argument and error out if we pass it
        # Inspect the forward method before we wrap the model with PEFT
        self.model_kwarg_keys = (
            inspect.signature(model.forward).parameters.keys()
            if not hasattr(model, "get_base_model")
            else inspect.signature(model.get_base_model().forward).parameters.keys()
        )

        if peft_config is not None or (is_peft_available() and isinstance(model, PeftModel)):
            model = prepare_peft_model(model, peft_config, args)

        # Processing class
        if processing_class is None:
            processing_class = AutoProcessor.from_pretrained(model.config._name_or_path)

        # Handle pad token for processors or tokenizers
        if isinstance(processing_class, ProcessorMixin):
            tokenizer = processing_class.tokenizer
        elif isinstance(processing_class, PreTrainedTokenizerBase):
            tokenizer = processing_class
        else:
            raise TypeError("The `processing_class` must be either a `PreTrainedTokenizerBase` or a `ProcessorMixin`")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.pad_token = tokenizer.pad_token
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.image_token = getattr(processing_class, "image_token", None)
        self.image_token_id = getattr(processing_class, "image_token_id", None)
        self.vision_start_token_id = getattr(model.config, "vision_start_token_id", None)
        self.vision_end_token_id = getattr(model.config, "vision_end_token_id", None)

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        self.reward_func_names = []
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
            if isinstance(reward_funcs[i], nn.Module):  # Use Module over PretrainedModel for compat w/ compiled models
                self.reward_func_names.append(reward_funcs[i].config._name_or_path.split("/")[-1])
            else:
                self.reward_func_names.append(reward_funcs[i].__name__)
        self.reward_funcs = reward_funcs

        # Reward weights
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(args.reward_weights)}) must match number of reward "
                    f"functions ({len(reward_funcs)})"
                )
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        if len(reward_processing_classes) != len(reward_funcs):
            raise ValueError(
                f"The number of reward processing classes ({len(reward_processing_classes)}) must match the number of "
                f"reward functions ({len(reward_funcs)})."
            )

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class

        self.reward_processing_classes = reward_processing_classes

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.min_p = args.min_p
        self.repetition_penalty = args.repetition_penalty
        self.use_transformers_paged = args.use_transformers_paged
        self.use_vllm = args.use_vllm
        self.vllm_mode = args.vllm_mode
        self.vllm_gpu_memory_utilization = args.vllm_gpu_memory_utilization  # only applies to colocation mode
        self.vllm_tensor_parallel_size = args.vllm_tensor_parallel_size  # only applies to colocation mode
        self.vllm_importance_sampling_correction = args.vllm_importance_sampling_correction
        self.vllm_importance_sampling_cap = args.vllm_importance_sampling_cap
        self.use_liger_loss = args.use_liger_loss
        self.loss_type = args.loss_type
        self.scale_rewards = args.scale_rewards
        self.importance_sampling_level = args.importance_sampling_level
        self.mask_truncated_completions = args.mask_truncated_completions
        self.top_entropy_quantile = args.top_entropy_quantile
        if self.use_liger_loss and self.top_entropy_quantile < 1.0:
            raise NotImplementedError(
                "Liger Kernels don't currently support masking token positions based on entropy."
            )
        if self.use_liger_loss and not self.importance_sampling_level == "token":
            raise NotImplementedError(
                "Liger Kernels currently only support token-level importance sampling. Please set"
                "`importance_sampling_level` to 'token'."
            )

        # Datasets
        self.shuffle_dataset = args.shuffle_dataset

        if (
            isinstance(train_dataset, IterableDataset)
            or isinstance(eval_dataset, IterableDataset)
            or (
                isinstance(eval_dataset, dict) and any(isinstance(ds, IterableDataset) for ds in eval_dataset.values())
            )
        ):
            # See https://github.com/huggingface/trl/issues/3213
            raise NotImplementedError(
                "Iterable datasets are not yet supported in GRPOTrainer. Please use a standard dataset instead."
            )

        # Multi-step
        self.num_iterations = args.num_iterations  # = 𝜇 in the GRPO paper
        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon
        # Tracks the number of iterations (forward + backward passes), including those within a grad accum cycle
        self._step = 0
        # Buffer the batch to reuse generated outputs across multiple updates. For more details, see
        # `_get_train_sampler` and `_prepare_inputs`.
        self._buffered_inputs = None

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        super().__init__(
            model=model,
            args=args,
            data_collator=identity,  # No data collation is needed in GRPO
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            # In Trainer, `training_step` scales the loss by `gradient_accumulation_steps` only if `compute_loss_func`
            # is None. For DAPO, loss scaling instead depends on the total number of completions tokens across the
            # global accumulated batch. To control scaling ourselves, we must disable Trainer’s built-in scaling. The
            # simplest (though a bit hacky) way is to set `compute_loss_func` to any non-None value, which bypasses
            # that behavior without rewriting `training_step`.
            compute_loss_func="non-None value to disable scaling",
        )

        # Reference model
        self.beta = args.beta
        if self.beta == 0.0:
            # If beta is 0.0, the reference model is not needed
            self.ref_model = None
        elif is_peft_model(model):
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None
        else:
            # For deepspeed, fsdp or non-distributed models, create a reference model from scratch
            config = AutoConfig.from_pretrained(model_id)
            architecture = getattr(transformers, config.architectures[0])
            self.ref_model = architecture.from_pretrained(model_id, **model_init_kwargs)

        # Disable dropout in the models
        if args.disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        # Liger loss
        if self.use_liger_loss:
            if not is_liger_kernel_available():
                raise ImportError(
                    "Liger is required to use `liger_loss` as the GRPO loss. Run `pip install liger-kernel`."
                )
            # redirect the model.module forward to the model forward to ensure pre-forward hooks are called
            self._forward_redirection = _ForwardRedirection()

            self.liger_grpo_loss = LigerFusedLinearGRPOLoss(
                beta=self.beta,
                epsilon_low=self.epsilon_low,
                epsilon_high=self.epsilon_high,
                temperature=self.temperature,
                use_ref_model=self.beta != 0.0,
                loss_type=self.loss_type,
                max_completion_length=self.max_completion_length,
            )

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0
        self.log_completions = args.log_completions
        self.wandb_log_unique_prompts = args.wandb_log_unique_prompts
        self.num_completions_to_print = args.num_completions_to_print
        # Keep logs sized to the generation batch to record only outputs from the latest model update.
        self._logs = {
            "images": deque(maxlen=args.generation_batch_size),
            "prompt": deque(maxlen=args.generation_batch_size),
            "completion": deque(maxlen=args.generation_batch_size),
            "rewards": defaultdict(lambda: deque(maxlen=args.generation_batch_size)),
            "advantages": deque(maxlen=args.generation_batch_size),
        }

        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)

        
        generation_kwargs = {
            "max_new_tokens": self.max_completion_length,
            "do_sample": True,
            "pad_token_id": tokenizer.pad_token_id,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "repetition_penalty": self.repetition_penalty,
            "cache_implementation": args.cache_implementation,
        }
        if args.generation_kwargs is not None:
            generation_kwargs.update(args.generation_kwargs)
        self.generation_config = GenerationConfig(**generation_kwargs)

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        # Add tags to the model
        self.model.add_model_tags(self._tag_names)

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            elif self.is_fsdp_enabled:
                self.ref_model = prepare_fsdp(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        if args.sync_ref_model:
            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                if self.is_deepspeed_enabled:
                    self.reward_funcs[i] = prepare_deepspeed(reward_func, self.accelerator)
                else:
                    # set device placement to True to make `prepare_model` move `reward_func` to device when using fsdp
                    self.reward_funcs[i] = self.accelerator.prepare_model(
                        reward_func, evaluation_mode=True, device_placement=True
                    )

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt", "image", "images", "point_cloud", "point_clouds"]

    # This method overrides `Trainer.get_train_dataloader` to support our custom batching strategy.
    # Instead of returning a standard per-step batch (i.e., `per_device_batch_size), our dataloader loads an
    # *generation* batch (i.e., `per_device_batch_size × steps_per_generation`). This allows us to generate completions
    # once every steps_per_generation step—rather than once per accumulation step—which is significantly more
    # efficient. The only change from the original implementation is multiplying the batch size by
    # `steps_per_generation`. Thus, `_prepare_inputs` is called with this *generation* batch, and it handles the
    # splitting internally.
    # Maintenance note: This method is a copy-paste of the original `Trainer.get_train_dataloader` with only one line
    # modification. As a result, some parts of the method aren't relevant to GRPO, but we keep them to stay one line
    # apart from the super method, ensuring easier maintenance in the future.
    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size * self.args.steps_per_generation,  # < this is the change
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = partial(
                seed_worker, num_workers=self.args.dataloader_num_workers, rank=self.args.process_index
            )

            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def _get_train_sampler(self, dataset: Optional[Dataset] = None) -> Sampler:
        # Returns a sampler that
        # 1. ensures each prompt is repeated across multiple processes. This guarantees that identical prompts are
        #    distributed to different GPUs, allowing rewards to be computed and normalized correctly within each prompt
        #    group. Using the same seed across processes ensures consistent prompt assignment, preventing discrepancies
        #    in group formation.
        # 2. repeats the batch multiple times to allow reusing generations across multiple updates. Refer to
        #    _prepare_inputs to see how the generations are stored and reused.

        # In the following figure, the values are the prompt indices. The first row shows the first sampled batch, the
        # second row shows the second sampled batch, and so on.
        #
        #                                      |   GPU 0  |   GPU 1  |
        #
        #                 global_step   step    <-───>  num_generations=2
        #                                       <-───────> per_device_train_batch_size=3
        #  grad_accum    ▲  ▲  0          0     0   0   1   1   2   2   <- Generate for the first `steps_per_generation` (prompts 0 to 11); store the completions; use the first slice to compute the loss
        #     =2         ▼  |  0          1     3   3   4   4   5   5   <- Take the stored generations and use the second slice to compute the loss
        #                   |
        #                   |  1          2     6   6   7   7   8   8   <- Take the stored generations and use the third slice to compute the loss
        #  steps_per_gen=4  ▼  1          3     9   9  10  10  11  11   <- Take the stored generations and use the fourth slice to compute the loss
        #
        #                      2          4    12  12  13  13  14  14   <- Generate for the second `steps_per_generation` (prompts 12 to 23); store the completions; use the first slice to compute the loss
        #                      2          5    15  15  16  16  17  17   <- Take the stored generations and use the second slice to compute the loss
        #                                          ...
        if dataset is None:
            dataset = self.train_dataset
        return RepeatSampler(
            data_source=dataset,
            mini_repeat_count=self.num_generations,
            batch_size=self.args.generation_batch_size // self.num_generations,
            repeat_count=self.num_iterations * self.args.steps_per_generation,
            shuffle=self.shuffle_dataset,
            seed=self.args.seed,
        )

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        # See _get_train_sampler for an explanation of the sampler.
        return RepeatSampler(
            data_source=eval_dataset,
            mini_repeat_count=self.num_generations,
            seed=self.args.seed,
        )

    @profiling_decorator
    def _get_last_hidden_state(
        self,
        unwrapped_model,
        input_ids,
        attention_mask,
        logits_to_keep,
        pixel_values=None,
        image_grid_thw=None,
        pixel_attention_mask=None,
        image_sizes=None,
    ):
        if is_peft_model(unwrapped_model):
            unwrapped_model = unwrapped_model.base_model.model

        # Build model inputs - check if the model supports logits_to_keep (some models and VLMs don't)
        model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

        # For Qwen models:
        if image_grid_thw is not None and pixel_values is not None:
            model_inputs["image_grid_thw"] = image_grid_thw
        # For Gemma, SmolVLM2, LLaVa-Next etc.:
        if pixel_values is not None:
            model_inputs["pixel_values"] = pixel_values
        # For SmolVLM2
        if pixel_attention_mask is not None:
            model_inputs["pixel_attention_mask"] = pixel_attention_mask
        # For LLaVa-Next
        if image_sizes is not None:
            model_inputs["image_sizes"] = image_sizes

        # Only add logits_to_keep if the model supports it
        if "logits_to_keep" in self.model_kwarg_keys:
            # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
            model_inputs["logits_to_keep"] = logits_to_keep + 1

        model_inputs["use_cache"] = False  # only used in generation; set False to suppress warnings

        last_hidden_state = unwrapped_model.model(**model_inputs).last_hidden_state
        # Exclude the last value: it corresponds to the next token pred
        last_hidden_state = last_hidden_state[:, :-1, :]  # (B, L-1, H)
        # Only keep the last logits_to_keep. For model that support logits_to_keep, this is a no-op.
        last_hidden_state = last_hidden_state[:, -logits_to_keep:, :]  # (B, logits_to_keep, H)
        return last_hidden_state

    def get_high_entropy_mask(self, entropies: torch.Tensor, mask: torch.Tensor, threshold: float) -> torch.Tensor:
        """
        Returns a binary mask identifying tokens whose entropy exceeds a given quantile threshold.

        Args:
            entropies (`torch.Tensor`):
                Tensor of shape (batch_size, seq_len) with per-token entropy values.
            mask (`torch.Tensor`):
                Binary mask of the same shape as `entropies`, where `1` indicates valid tokens and `0` padding.
            threshold (`float`):
                Quantile threshold between `0.0` and `1.0` to select high-entropy tokens.

        Returns:
            `torch.Tensor`:
                Boolean mask of shape (batch_size, seq_len), where `True` indicates tokens with entropy >= threshold
                and `False` otherwise.
        """
        local = entropies[mask.bool()].float()

        # Use a negative pad_value as a sentinel because entropy values are always >= 0.
        # This guarantees that the sentinel cannot collide with any real entropy value.
        pad_value = -1e9

        # Pad across processes so that every rank has the same tensor length
        padded = self.accelerator.pad_across_processes(local, dim=0, pad_index=pad_value)
        gathered = self.accelerator.gather(padded)

        # Drop sentinel values (safe because no entropy can be negative)
        gathered = gathered[gathered != pad_value]

        if gathered.numel() == 0:
            return torch.zeros_like(entropies, dtype=torch.bool)

        entropy_threshold = torch.quantile(gathered, threshold)
        masked_entropies = entropies * mask.float()
        entropy_mask = masked_entropies >= entropy_threshold
        return entropy_mask & mask.bool()  # ensure padding tokens are always masked out

    @profiling_decorator
    def _get_per_token_logps_and_entropies(
        self,
        model,
        point_clouds,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size=None,
        compute_entropy=False,
        pixel_values=None,
        image_grid_thw=None,
        num_images=None,
        pixel_attention_mask=None,
        image_sizes=None,
    ) -> dict[str, Optional[torch.Tensor]]:
        """Compute log-probs and (optionally) entropies for each token."""
        batch_size = batch_size or input_ids.size(0)  # Chunk inputs into smaller batches to reduce memory peak
        all_logps = []
        all_entropies = []
        for start in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[start : start + batch_size]
            attention_mask_batch = attention_mask[start : start + batch_size]

            # Build model inputs - check if the model supports logits_to_keep (some models and VLMs don't)
            model_inputs = {"input_ids": input_ids_batch, "attention_mask": attention_mask_batch}
            if image_grid_thw is not None and pixel_values is not None:
                rows_per_image = image_grid_thw.prod(dim=-1)
                rows_per_sample = torch.split(rows_per_image, num_images)
                rows_per_sample = torch.stack([s.sum() for s in rows_per_sample])
                cum_rows = torch.cat([torch.tensor([0], device=rows_per_sample.device), rows_per_sample.cumsum(0)])
                row_start, row_end = cum_rows[start].item(), cum_rows[start + batch_size].item()
                model_inputs["pixel_values"] = pixel_values[row_start:row_end]
                cum_imgs = torch.tensor([0] + num_images).cumsum(0)
                img_start, img_end = cum_imgs[start], cum_imgs[start + batch_size]
                model_inputs["image_grid_thw"] = image_grid_thw[img_start:img_end]
            elif pixel_values is not None:
                model_inputs["pixel_values"] = pixel_values[start : start + batch_size]
            if pixel_attention_mask is not None:
                model_inputs["pixel_attention_mask"] = pixel_attention_mask[start : start + batch_size]
            if image_sizes is not None:
                model_inputs["image_sizes"] = image_sizes[start : start + batch_size]

            model_inputs["point_cloud"] = point_clouds

            # Only add logits_to_keep if the model supports it
            if "logits_to_keep" in self.model_kwarg_keys:
                # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
                model_inputs["logits_to_keep"] = logits_to_keep + 1

            model_inputs["use_cache"] = False  # only used in generation; set False to suppress warnings

            logits = model(**model_inputs).logits
            # Exclude the last value: it corresponds to the next token pred
            logits = logits[:, :-1, :]  # (B, L-1, H)
            # Only keep the last logits_to_keep. For model that support logits_to_keep, this is a no-op.
            logits = logits[:, -logits_to_keep:, :]  # (B, logits_to_keep, H)
            # Divide logits by sampling temperature.
            # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
            logits = logits / self.temperature

            completion_ids = input_ids_batch[:, -logits_to_keep:]
            logps = selective_log_softmax(logits, completion_ids)  # compute logprobs
            all_logps.append(logps)

            if compute_entropy:
                with torch.no_grad():
                    entropies = entropy_from_logits(logits)
                all_entropies.append(entropies)

        logps = torch.cat(all_logps, dim=0)
        entropies = torch.cat(all_entropies, dim=0) if compute_entropy else None
        return logps, entropies

    @profiling_decorator
    def _prepare_inputs(
        self, generation_batch: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        # Prepares inputs for model training/evaluation by managing completion generation and batch handling.
        # During training:
        #   - Receives the local generation batch (Per-GPU batch size × steps per generation)
        #     from the modified training dataloader instead of the standard local batch
        #   - Generates completions once for the entire generation batch and splits it into batches of size
        #     `per_device_train_batch_size`
        #   - Buffers these completions and returns the appropriate slice for the current accumulation step
        #   - Optimizes by regenerating completions only periodically (every steps_per_generation * num_iterations)
        # During evaluation:
        #   - The input is treated as a standard local batch (no accumulation, no multiple iterations)
        #   - Completions are generated for each batch without buffering or reuse
        # Returns a single local batch in both cases.

        mode = "train" if self.model.training else "eval"
        if mode == "train":
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self._step % generate_every == 0 or self._buffered_inputs is None:
                # self._buffered_inputs=None can occur when resuming from a checkpoint
                generation_batch = self._generate_and_score_completions(generation_batch)
                generation_batch = split_pixel_values_by_grid(generation_batch)


                # ==================== DEBUGGING BLOCK ====================
                # print("\n" + "="*20 + " DEBUG: Before Shuffle " + "="*20)
                # if torch.cuda.current_device() == 0:
                #     for key, value in generation_batch.items():
                #         # 检查 value 是否是 PyTorch Tensor
                #         if isinstance(value, torch.Tensor):
                #             print(f"Key: {key:<25} | Type: torch.Tensor, Shape: {value.shape}")
                #         # 检查 value 是否是 list
                #         elif isinstance(value, list):
                #             # 首先打印 list 本身的信息
                #             print(f"Key: {key:<25} | Type: list, Length: {len(value)}")
                            
                #             # 如果 list 不为空，则深入检查第一个元素
                #             if len(value) > 0:
                #                 first_element = value[0]
                #                 element_type = type(first_element)
                                
                #                 # 检查第一个元素是否也是 Tensor
                #                 if isinstance(first_element, torch.Tensor):
                #                     print(f"{'':<28} |--> Element Type: torch.Tensor, Element Shape: {first_element.shape}")
                #                 # 检查第一个元素是否也是 list
                #                 elif isinstance(first_element, list):
                #                     print(f"{'':<28} |--> Element Type: list, Inner List Length: {len(first_element)}")
                #                 # 其他所有类型（如 int, str, dict 等）
                #                 else:
                #                     print(f"{'':<28} |--> Element Type: {element_type.__name__}")
                #             else:
                #                 print(f"{'':<28} |--> (List is empty)")
                                
                #         # 其他所有类型
                #         else:
                #             print(f"Key: {key:<25} | Type: {type(value).__name__}, Value: {value}")

                #     print("="* (40 + len(" DEBUG: Before Shuffle ") + 1) + "\n")
                # exit()
                # =========================================================


                generation_batch = shuffle_sequence_dict(generation_batch)
                generation_batches = split_tensor_dict(generation_batch, self.args.steps_per_generation)
                self._buffered_inputs = [unsplit_pixel_values_by_grid(batch) for batch in generation_batches]
            inputs = self._buffered_inputs[self._step % self.args.steps_per_generation]
            self._step += 1
        else:
            # In evaluation, there is neither batch grouping for generation, nor multiple iterations, hence
            # local generation batch == local eval batch
            inputs = self._generate_and_score_completions(generation_batch)
        return inputs


    @profiling_decorator
    def _calculate_rewards(self, inputs, prompts, completions, completion_ids, stage):
        """
        计算当前轮次的奖励。
        参数:
            prompts: List[str]
            completions: List[Dict]  <-- 改为直接接收字典列表
            stage: int (1 or 2)
        """
        device = self.accelerator.device
        
        # 1. 直接获取唯一的奖励函数
        reward_func = self.reward_funcs[0]

        # 2. 准备 kwargs
        # 排除掉已经被显式传递的参数
        keys = [key for key in inputs[0] if key not in ["prompt", "completion", "completion_ids"]]
        reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
        
        # 注入 stage 和 state
        reward_kwargs["stage"] = stage 
        reward_kwargs["trainer_state"] = self.state
        
        # 3. 直接调用奖励函数
        # 注意：你的 reward_func 内部应该能处理 completions 是 List[Dict] 的情况
        output_reward_func = reward_func(
            prompts=prompts, 
            completions=completions, 
            completion_ids=completion_ids, 
            **reward_kwargs
        )
        
        # 4. 转换为 Tensor 并移动到 Device
        rewards = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # 5. (多GPU) Gather 操作
        gathered_rewards = self.accelerator.gather(rewards)
        
        return gathered_rewards

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        # =================================================================================================
        # STAGE 1: 第一轮生成 (Turn 1 Generation)
        # =================================================================================================
        
        # --- 1.1 [Turn 1] 数据准备与 Prompt 构建 ---
        prompts_text_t1 = []
        images_list_t1 = []
        point_clouds_list = []
        user_msgs_cache = [] # 用于缓存第一轮的用户输入，供第二轮拼接历史使用

        for example in inputs:
            # 深拷贝防止修改原数据
            import copy
            user_message = copy.deepcopy(example["messages"][0])
            user_msgs_cache.append(user_message) # 存下来给 STAGE 2 用

            # 提取图像
            images = example.get("images") or ([example.get("image")] if example.get("image") is not None else None)

            if not isinstance(images, list) and images:
                images = [images]
            
            # 提取点云
            point_cloud = example.get("point_cloud")
            point_clouds_list.append(point_cloud)

            # 构建 Chat Template
            temp_example_for_template = {"prompt": [user_message]}

            formatted_prompt = maybe_apply_chat_template(
                temp_example_for_template, self.processing_class
            )["prompt"]
            prompts_text_t1.append(formatted_prompt)
            
            # 收集图像列表 (对应 Batch 顺序)
            if images:
                images_list_t1.append(images)

        # 处理点云 Tensor 堆叠
        pcs_tensor = [torch.tensor(pc, dtype=torch.float32, device=device) for pc in point_clouds_list]
        point_clouds = torch.stack(pcs_tensor, dim=0).reshape(-1, 1024, 3)

        # --- 1.2 [Turn 1] Tokenize ---
        kwargs_t1 = {}
        if images_list_t1:
            kwargs_t1["images"] = images_list_t1

        prompt_inputs_t1 = self.processing_class(
            text=prompts_text_t1,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
            **kwargs_t1,
        )
        prompt_inputs_t1['point_cloud'] = point_clouds
        # 发送到设备
        prompt_inputs_t1 = super()._prepare_inputs(prompt_inputs_t1)
        
        prompt_ids_t1 = prompt_inputs_t1["input_ids"]
        prompt_mask_t1 = prompt_inputs_t1["attention_mask"]

        # --- 1.3 [Turn 1] 执行生成 ---
        with (
            profiling_context(self, "transformers.generate_t1"),
            unwrap_model_for_generation(
                self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
            ) as unwrapped_model,
            torch.no_grad(),
            FSDP.summon_full_params(self.model_wrapped, recurse=False) if self.is_fsdp_enabled else nullcontext(),
        ):
            # prompt_completion_ids_t1 包含 [Prompt1 + Gen1]
            prompt_completion_ids_t1 = unwrapped_model.generate(
                **prompt_inputs_t1, generation_config=self.generation_config, disable_compile=True
            )

        # --- 1.4 [Turn 1] 后处理：提取 Gen1 & 构建 Mask ---
        # 提取生成部分
        prompt_length_t1 = prompt_ids_t1.size(1) 
        prompt_ids_t1 = prompt_completion_ids_t1[:, :prompt_length_t1]
        completion_ids_t1 = prompt_completion_ids_t1[:, prompt_length_t1:]
        
        # 构建 EOS Mask
        is_eos_t1 = completion_ids_t1 == self.eos_token_id
        eos_idx_t1 = torch.full((is_eos_t1.size(0),), is_eos_t1.size(1), dtype=torch.long, device=device)
        eos_idx_t1[is_eos_t1.any(dim=1)] = is_eos_t1.int().argmax(dim=1)[is_eos_t1.any(dim=1)]
        sequence_indices_t1 = torch.arange(is_eos_t1.size(1), device=device).expand(is_eos_t1.size(0), -1)
        completion_mask_t1 = (sequence_indices_t1 <= eos_idx_t1.unsqueeze(1)).int()

        # 处理被截断的情况
        if self.mask_truncated_completions:
            truncated_completions_t1 = ~is_eos_t1.any(dim=1)
            completion_mask_t1 = completion_mask_t1 * (~truncated_completions_t1).unsqueeze(1).int()

        # 解码文本，用于构建第二轮历史
        completions_text_t1 = self.processing_class.batch_decode(completion_ids_t1, skip_special_tokens=True)
        # 结构化为 Assistant Message 列表
        completions_struct_t1 = [{"role": "assistant", "content": text} for text in completions_text_t1]

        # =================================================================================================
        # STAGE 2: 第二轮生成 (Turn 2 Generation)
        # =================================================================================================

        # # --- 2.1 [Turn 2] 数据准备与 Prompt 构建 (拼接历史) ---
        prompts_text_t2 = []
        images_list_t2 = [] # 这里我们直接复用 images_list_t1，因为图片在历史里是一样的，但为了逻辑严谨重新遍历一遍
        
        # 定义第二轮的用户 Prompt
        user_msg_t2_content = "Please perform a second round of refinement based on the visual features and the provided predictions from the previous round. Strictly output your final, precise predictions in the specified JSON format below."
        user_msg_t2_struct = {"role": "user", "content": user_msg_t2_content}

        for i, example in enumerate(inputs):
            # 1. 获取第一轮的用户输入 (缓存的)
            msg_u1 = user_msgs_cache[i]
            # 2. 获取第一轮的模型输出 (刚刚生成的)
            msg_a1 = completions_struct_t1[i]
            # 3. 定义第二轮用户输入
            msg_u2 = user_msg_t2_struct

            # 4. 拼接完整对话历史 [U1, A1, U2]
            full_history = [msg_u1, msg_a1, msg_u2]
            
            # 5. 应用模板
            temp_example_t2 = {"prompt": full_history}
            formatted_prompt_t2 = maybe_apply_chat_template(
                temp_example_t2, self.processing_class
            )["prompt"]
            prompts_text_t2.append(formatted_prompt_t2)

            # 6. 图像处理 (复用逻辑)
            images = example.get("images") or ([example.get("image")] if example.get("image") is not None else None)
            if not isinstance(images, list) and images:
                images = [images]
            if images:
                images_list_t2.append(images)

        # --- 2.2 [Turn 2] Tokenize (注意：这次是对包含了历史的 Prompt 进行编码) ---
        kwargs_t2 = {}
        if images_list_t2:
            kwargs_t2["images"] = images_list_t2
        
        prompt_inputs_t2 = self.processing_class(
            text=prompts_text_t2,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
            **kwargs_t2,
        )
        prompt_inputs_t2['point_cloud'] = point_clouds # 点云是一样的，复用 Tensor
        prompt_inputs_t2 = super()._prepare_inputs(prompt_inputs_t2)

        prompt_ids_t2 = prompt_inputs_t2["input_ids"]
        prompt_mask_t2 = prompt_inputs_t2["attention_mask"]
        prompt_length_t2 = prompt_ids_t2.size(1) # 此时长度包含 [P1 + G1 + P2]

        # --- 2.3 [Turn 2] 执行生成 ---
        with (
            profiling_context(self, "transformers.generate_t2"),
            unwrap_model_for_generation(
                self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
            ) as unwrapped_model,
            torch.no_grad(),
            FSDP.summon_full_params(self.model_wrapped, recurse=False) if self.is_fsdp_enabled else nullcontext(),
        ):
            # prompt_completion_ids_t2 包含 [P1, G1, P2, G2]
            prompt_completion_ids_t2 = unwrapped_model.generate(
                **prompt_inputs_t2, generation_config=self.generation_config, disable_compile=True
            )

        # --- 2.4 [Turn 2] 后处理：提取 Gen2 & 构建 Mask ---
        completion_ids_t2 = prompt_completion_ids_t2[:, prompt_length_t2:]

        # 构建 EOS Mask
        is_eos_t2 = completion_ids_t2 == self.eos_token_id
        eos_idx_t2 = torch.full((is_eos_t2.size(0),), is_eos_t2.size(1), dtype=torch.long, device=device)
        eos_idx_t2[is_eos_t2.any(dim=1)] = is_eos_t2.int().argmax(dim=1)[is_eos_t2.any(dim=1)]
        sequence_indices_t2 = torch.arange(is_eos_t2.size(1), device=device).expand(is_eos_t2.size(0), -1)
        completion_mask_t2 = (sequence_indices_t2 <= eos_idx_t2.unsqueeze(1)).int()

        if self.mask_truncated_completions:
            truncated_completions_t2 = ~is_eos_t2.any(dim=1)
            completion_mask_t2 = completion_mask_t2 * (~truncated_completions_t2).unsqueeze(1).int()

        # 解码第二轮文本
        completions_text_t2 = self.processing_class.batch_decode(completion_ids_t2, skip_special_tokens=True)
        # 结构化 (虽然这里不用拼接下一轮了，但 Reward 可能需要)
        completions_struct_t2 = [{"role": "assistant", "content": text} for text in completions_text_t2]

        # =================================================================================================
        # STAGE 3: 辅助 Logps 计算 (Old Policy & Reference Model)
        # =================================================================================================
        # 我们需要为 T1 和 T2 分别计算 logps，因为它们是两个独立的 tensor 序列

        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size
        num_images_t1 = [len(img_list) for img_list in images_list_t1] if images_list_t1 else None
        num_images_t2 = [len(img_list) for img_list in images_list_t2] if images_list_t2 else None
        
        # --- 3.1 [Turn 1] Logps ---
        attention_mask_t1 = torch.cat([prompt_mask_t1, completion_mask_t1], dim=1)
        logits_to_keep_t1 = completion_ids_t1.size(1)
        
        # Old Logps T1
        old_per_token_logps_t1 = None
        generate_every = self.args.steps_per_generation * self.num_iterations
        if self.args.gradient_accumulation_steps % generate_every != 0 or (
            self.use_vllm and self.vllm_importance_sampling_correction
        ):
            old_per_token_logps_t1, _ = self._get_per_token_logps_and_entropies(
                self.model, point_clouds, prompt_completion_ids_t1, attention_mask_t1, logits_to_keep_t1, batch_size,
                pixel_values=prompt_inputs_t1.get("pixel_values"), image_grid_thw=prompt_inputs_t1.get("image_grid_thw"),
                num_images=num_images_t1, pixel_attention_mask=prompt_inputs_t1.get("pixel_attention_mask"),
                image_sizes=prompt_inputs_t1.get("image_sizes"),
            )
            
        # Ref Logps T1
        ref_per_token_logps_t1 = None
        if self.beta != 0.0:
            model_to_use = self.ref_model if self.ref_model else self.model
            with (nullcontext() if self.ref_model else self.accelerator.unwrap_model(self.model).disable_adapter()):
                ref_per_token_logps_t1, _ = self._get_per_token_logps_and_entropies(
                    model_to_use, point_clouds, prompt_completion_ids_t1, attention_mask_t1, logits_to_keep_t1, batch_size,
                    pixel_values=prompt_inputs_t1.get("pixel_values"), image_grid_thw=prompt_inputs_t1.get("image_grid_thw"),
                    num_images=num_images_t1, pixel_attention_mask=prompt_inputs_t1.get("pixel_attention_mask"),
                    image_sizes=prompt_inputs_t1.get("image_sizes"),
                )

        # --- 3.2 [Turn 2] Logps ---
        attention_mask_t2 = torch.cat([prompt_mask_t2, completion_mask_t2], dim=1)
        logits_to_keep_t2 = completion_ids_t2.size(1)

        # Old Logps T2
        old_per_token_logps_t2 = None
        if self.args.gradient_accumulation_steps % generate_every != 0 or (
            self.use_vllm and self.vllm_importance_sampling_correction
        ):
            old_per_token_logps_t2, _ = self._get_per_token_logps_and_entropies(
                self.model, point_clouds, prompt_completion_ids_t2, attention_mask_t2, logits_to_keep_t2, batch_size,
                pixel_values=prompt_inputs_t2.get("pixel_values"), image_grid_thw=prompt_inputs_t2.get("image_grid_thw"),
                num_images=num_images_t2, pixel_attention_mask=prompt_inputs_t2.get("pixel_attention_mask"),
                image_sizes=prompt_inputs_t2.get("image_sizes"),
            )

        # Ref Logps T2
        ref_per_token_logps_t2 = None
        if self.beta != 0.0:
            model_to_use = self.ref_model if self.ref_model else self.model
            with (nullcontext() if self.ref_model else self.accelerator.unwrap_model(self.model).disable_adapter()):
                ref_per_token_logps_t2, _ = self._get_per_token_logps_and_entropies(
                    model_to_use, point_clouds, prompt_completion_ids_t2, attention_mask_t2, logits_to_keep_t2, batch_size,
                    pixel_values=prompt_inputs_t2.get("pixel_values"), image_grid_thw=prompt_inputs_t2.get("image_grid_thw"),
                    num_images=num_images_t2, pixel_attention_mask=prompt_inputs_t2.get("pixel_attention_mask"),
                    image_sizes=prompt_inputs_t2.get("image_sizes"),
                )

        # =================================================================================================
        # STAGE 4: Reward & Advantage 计算
        # =================================================================================================
        
        # 准备 Token List 供 reward 使用 (Avoid re-tokenization)
        completion_ids_list_t1 = [row[mask.bool()].tolist() for row, mask in zip(completion_ids_t1, completion_mask_t1)]
        completion_ids_list_t2 = [row[mask.bool()].tolist() for row, mask in zip(completion_ids_t2, completion_mask_t2)]

        # --- 4.1 计算 Rewards ---
        # 注意：这里假设 _calculate_rewards 可以接收 stage 参数，或者你可以根据 inputs 区分
        # 如果你的 reward function 是通用的，直接传即可。
        # 为了区分，我这里传入了 structure 列表，你可以根据 structure 里的内容判断是第一轮还是第二轮
        rewards_t1 = self._calculate_rewards(
            inputs, 
            prompts=prompts_text_t1, 
            completions=completions_struct_t1, 
            completion_ids=completion_ids_list_t1,
            stage=1
        )

        # Turn 2: 直接传入 completions_struct_t2 (List[Dict])
        rewards_t2 = self._calculate_rewards(
            inputs, 
            prompts=prompts_text_t2, 
            completions=completions_struct_t2, 
            completion_ids=completion_ids_list_t2,
            stage=2
        )
        print(f"Rewards T1: {rewards_t1}")
        print(f"Rewards T2: {rewards_t2}")

        # --- 4.2 计算 Advantages (Turn 1) ---
        mean_grouped_rewards_t1 = rewards_t1.view(-1, self.num_generations).mean(dim=1).repeat_interleave(self.num_generations, dim=0)
        advantages_t1 = rewards_t1 - mean_grouped_rewards_t1
        
        if self.scale_rewards in ["group", "none"]:
            std_rewards_t1 = rewards_t1.view(-1, self.num_generations).std(dim=1).repeat_interleave(self.num_generations, dim=0)
        elif self.scale_rewards == "batch":
            std_rewards_t1 = rewards_t1.std().expand_as(rewards_t1)
        else:
            std_rewards_t1 = torch.ones_like(rewards_t1)
            
        if self.scale_rewards != "none":
            advantages_t1 = advantages_t1 / (std_rewards_t1 + 1e-4)

        # --- 4.3 计算 Advantages (Turn 2) ---
        mean_grouped_rewards_t2 = rewards_t2.view(-1, self.num_generations).mean(dim=1).repeat_interleave(self.num_generations, dim=0)
        advantages_t2 = rewards_t2 - mean_grouped_rewards_t2
        
        if self.scale_rewards in ["group", "none"]:
            std_rewards_t2 = rewards_t2.view(-1, self.num_generations).std(dim=1).repeat_interleave(self.num_generations, dim=0)
        elif self.scale_rewards == "batch":
            std_rewards_t2 = rewards_t2.std().expand_as(rewards_t2)
        else:
            std_rewards_t2 = torch.ones_like(rewards_t2)
            
        if self.scale_rewards != "none":
            advantages_t2 = advantages_t2 / (std_rewards_t2 + 1e-4)

        # 切片获取当前进程数据
        process_slice = slice(self.accelerator.process_index * len(inputs), (self.accelerator.process_index + 1) * len(inputs))
        advantages_t1 = advantages_t1[process_slice]
        advantages_t2 = advantages_t2[process_slice]

        # =================================================================================================
        # STAGE 5: 输出打包
        # =================================================================================================
        
        num_items_in_batch_t1 = self.accelerator.gather(completion_mask_t1.sum(1)).sum()
        num_items_in_batch_t2 = self.accelerator.gather(completion_mask_t2.sum(1)).sum()

        # print("" * 20 + " DEBUG: Generation Shapes " + "*" * 20)
        # print(f"pixel values t1 shape: {prompt_inputs_t1.get('pixel_values').shape}")
        # print(f"pixel values t2 shape: {prompt_inputs_t2.get('pixel_values').shape}")
        # print(f"=" * (40 + len(" DEBUG: Generation Shapes ") + 1))

        output = {
            # Turn 1 Data
            "prompt_ids_t1": prompt_ids_t1,
            "prompt_mask_t1": prompt_mask_t1,
            "completion_ids_t1": completion_ids_t1,
            "completion_mask_t1": completion_mask_t1,
            "advantages_t1": advantages_t1,
            "num_items_in_batch_t1": num_items_in_batch_t1,
            "logits_to_keep_t1": [logits_to_keep_t1] * self.num_generations,
            
            # Turn 2 Data
            "prompt_ids_t2": prompt_ids_t2,
            "prompt_mask_t2": prompt_mask_t2,
            "completion_ids_t2": completion_ids_t2,
            "completion_mask_t2": completion_mask_t2,
            "advantages_t2": advantages_t2,
            "num_items_in_batch_t2": num_items_in_batch_t2,
            "logits_to_keep_t2": [logits_to_keep_t2] * self.num_generations,

            # Image Data
            "pixel_values": prompt_inputs_t1.get("pixel_values"),
            "image_grid_thw": prompt_inputs_t1.get("image_grid_thw"),
            "num_images": num_images_t1,
            "pixel_attention_mask": prompt_inputs_t1.get("pixel_attention_mask"),
            "image_sizes": prompt_inputs_t1.get("image_sizes"),

            # Shared Data
            "point_cloud": point_clouds, 
        }


        if old_per_token_logps_t1 is not None: output["old_per_token_logps_t1"] = old_per_token_logps_t1
        if ref_per_token_logps_t1 is not None: output["ref_per_token_logps_t1"] = ref_per_token_logps_t1
        if old_per_token_logps_t2 is not None: output["old_per_token_logps_t2"] = old_per_token_logps_t2
        if ref_per_token_logps_t2 is not None: output["ref_per_token_logps_t2"] = ref_per_token_logps_t2

        # 简单打点 Log
        # if mode == "train":
        #     total_tokens = (prompt_mask_t1.sum() + completion_mask_t1.sum() + prompt_mask_t2.sum() + completion_mask_t2.sum())
        #     self.state.num_input_tokens_seen += self.accelerator.gather(total_tokens).sum().item()
        
        self._metrics[mode]["reward_t1"].append(rewards_t1.mean().item())
        self._metrics[mode]["reward_t2"].append(rewards_t2.mean().item())
        # self._logs["prompt"].extend(gather_object(prompts_text_t2)) # 记录完整的 T2 Prompt
        # self._logs["completion"].extend(gather_object(completions_text_t2)) # 记录 T2 Completion

        return output

    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        if self.use_liger_loss:
            # Compute the loss using the liger grpo loss
            unwrapped_model = self.accelerator.unwrap_model(model)
            return self._forward_redirection(model, unwrapped_model, self.compute_liger_loss, unwrapped_model, inputs)
        else:
            return self._compute_loss(model, inputs)

    def _compute_loss(self, model, inputs):
        pad_token_id = self.processing_class.tokenizer.pad_token_id

        # =================================================================================================
        # 2. 数据准备 (Data Preparation)
        # =================================================================================================
        
        # --- T1 ---
        ids_t1 = torch.cat([inputs["prompt_ids_t1"], inputs["completion_ids_t1"]], dim=1)
        # Prompt=1, Gen=1 (用于Attention)
        full_mask_t1 = torch.cat([inputs["prompt_mask_t1"], inputs["completion_mask_t1"]], dim=1)
        # Prompt=0, Gen=1 (用于Loss计算)
        # 注意：inputs["completion_mask_t1"] 仅包含生成部分的 mask (全1，可能尾部有0)
        # gen_mask_t1 = inputs["completion_mask_t1"]

        # --- T2 ---
        ids_t2 = torch.cat([inputs["prompt_ids_t2"], inputs["completion_ids_t2"]], dim=1)
        full_mask_t2 = torch.cat([inputs["prompt_mask_t2"], inputs["completion_mask_t2"]], dim=1)
        # gen_mask_t2 = inputs["completion_mask_t2"]

        batch_size = ids_t1.size(0)
        device = f"cuda:{torch.cuda.current_device()}"

        # =================================================================================================
        # 3. Left Padding & Batch Construction (左侧 Padding 与拼接)
        # =================================================================================================
        
        # 对于 Input IDs，为了让生成部分对齐到最右侧，必须使用 Left Padding
        max_len = max(ids_t1.size(1), ids_t2.size(1))

        def left_pad_tensor(t, target_len, val):
            curr = t.size(1)
            if curr >= target_len: return t
            return torch.nn.functional.pad(t, (target_len - curr, 0), value=val)

        # 1. Input IDs: 左补 Pad
        ids_t1_pad = left_pad_tensor(ids_t1, max_len, pad_token_id)
        ids_t2_pad = left_pad_tensor(ids_t2, max_len, pad_token_id)
        combined_input_ids = torch.cat([ids_t1_pad, ids_t2_pad], dim=0)

        # 2. Attention Mask: 左补 0
        att_t1_pad = left_pad_tensor(full_mask_t1, max_len, 0)
        att_t2_pad = left_pad_tensor(full_mask_t2, max_len, 0)
        combined_attention_mask = torch.cat([att_t1_pad, att_t2_pad], dim=0)

        # =================================================================================================
        # 4. Vision Data Handling (视觉数据拼接)
        # =================================================================================================
        
        # 这里的逻辑是：input_ids 翻倍了，共享的图片也需要翻倍
        pv = inputs["pixel_values"]
        combined_pixel_values = torch.cat([pv, pv], dim=0) if pv is not None else None

        grid = inputs["image_grid_thw"]
        combined_image_grid_thw = torch.cat([grid, grid], dim=0) if grid is not None else None

        pc = inputs["point_cloud"]
        combined_point_cloud = torch.cat([pc, pc], dim=0) if pc is not None else None

        # =================================================================================================
        # 5. Single Forward (单次前向)
        # =================================================================================================
        
        # 计算需要保留的 Logits 长度
        # 逻辑：我们需要计算 Loss 的部分是 completion_ids。
        # 为了预测 completion 的第一个 token，我们需要它前面的那个 token 的 logit。
        # 所以我们需要保留 completion_len + 1 个 logits。

        len_gen_t1 = inputs["completion_ids_t1"].size(1)
        len_gen_t2 = inputs["completion_ids_t2"].size(1)
        
        # 取两者最大值，并 +1 (为了 Shift)
        max_gen_len = max(len_gen_t1, len_gen_t2)

        logits_to_keep = max_gen_len + 1

        model_inputs = {
            "input_ids": combined_input_ids,
            "attention_mask": combined_attention_mask,
            "pixel_values": combined_pixel_values,
            "image_grid_thw": combined_image_grid_thw,
            "point_cloud": combined_point_cloud,
            "use_cache": False,
            "logits_to_keep": logits_to_keep # 【修正】保留 N+1 个
        }

        outputs = model(**model_inputs)
        logits = outputs.logits # [2*B, logits_to_keep, Vocab]
        logits = logits[:, :-1, :]

        # =================================================================================================
        # 6. Split & Compute Loss (拆分与计算)
        # =================================================================================================
        
        logits_t1 = logits[0, -max_gen_len:].unsqueeze(0)
        logits_t2 = logits[1, -max_gen_len:].unsqueeze(0)

        # 准备 Labels: 取 input_ids 的最后 max_gen_len 个
        # 因为做了 Left Padding，有效内容都在最右侧，直接取最后 N 个即为生成的 Token
        labels_t1 = ids_t1_pad[:, -max_gen_len:]
        labels_t2 = ids_t2_pad[:, -max_gen_len:]

        def loss_mask(max_gen_len, gen_len, label):
            pad_mask = (label == pad_token_id).int()
            
            if gen_len == max_gen_len:
                mask = torch.ones((max_gen_len,), dtype=torch.int, device=device)
            else:
                mask_0 = torch.zeros((max_gen_len - gen_len,), dtype=torch.int, device=device)
                mask_1 = torch.ones((gen_len,), dtype=torch.int, device=device)
                mask = torch.cat([mask_0, mask_1], dim=0)

            return mask * (1 - pad_mask)

            
        loss_mask_t1 = loss_mask(max_gen_len, len_gen_t1, labels_t1)
        loss_mask_t2 = loss_mask(max_gen_len, len_gen_t2, labels_t2)

        # 定义子函数
        def compute_masked_loss(
            curr_logits, curr_labels, loss_mask, 
            advantages, num_items_in_batch, 
            old_logps, ref_logps
        ):
            """
            curr_logits: [B, max_gen_len, V]
            curr_labels: [B, max_gen_len]
            orig_gen_mask: [B, real_gen_len] (T1较短，T2较长)
            """
            
            shift_logits = curr_logits # [B, max_gen_len, V]
            shift_labels = curr_labels            # [B, max_gen_len]
            shift_mask = loss_mask             # [B, max_gen_len]

            # if torch.cuda.current_device() == 0:
            #     print(f"shift_logits shape: {shift_logits.shape}")
            #     print(f"shift_labels shape: {shift_labels.shape}")
            #     print(f"shift_mask shape: {shift_mask.shape}")
            #     print(f"advantages shape: {advantages.shape}")
            

            # 3. 计算 Logps
            shift_logits = shift_logits / self.temperature
            per_token_logps = selective_log_softmax(shift_logits, shift_labels) # [B, max_gen_len]

            # 4. Old Logps 对齐
            # old_logps 是 Rollout 时算的，长度是 real_gen_len。
            # 需要左补 0 对齐到 max_gen_len
            # target_len = per_token_logps.size(1)

            old_logps = per_token_logps.detach()
            
            # if old_logps is None:
            #     old_logps = per_token_logps.detach()
            # else:
            #     if old_logps.size(1) < target_len:
            #         old_logps = left_pad_tensor(old_logps, target_len, 0.0)
            #     elif old_logps.size(1) > target_len:
            #         # 防御性截断 (取右侧，因为是左Padding)
            #         old_logps = old_logps[:, -target_len:]

            # 5. GRPO Loss
            log_ratio = per_token_logps - old_logps
            coef_1 = torch.exp(log_ratio)
            coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
            
            per_token_loss = -torch.min(coef_1 * advantages.unsqueeze(1), coef_2 * advantages.unsqueeze(1))
            
            # KL Penalty
            # if self.beta != 0.0 and ref_logps is not None:
            #      if ref_logps.size(1) < target_len:
            #          ref_logps = left_pad_tensor(ref_logps, target_len, 0.0)
            #      elif ref_logps.size(1) > target_len:
            #          ref_logps = ref_logps[:, -target_len:]
                 
            #      kl = torch.exp(ref_logps - per_token_logps) - (ref_logps - per_token_logps) - 1
            #      per_token_loss = per_token_loss + self.beta * kl

            # 6. Apply Mask (把 Pad 部分的 Loss 抹零)
            valid_loss = per_token_loss * shift_mask

            normalizer = num_items_in_batch / self.accelerator.num_processes
            loss = valid_loss.sum() / normalizer
            
            # 7. Normalize
            # if self.loss_type == "dapo":
            #     normalizer = num_items_in_batch / self.accelerator.num_processes
            #     loss = valid_loss.sum() / normalizer
            # else:
            #     # sum(-1) 求每个样本的 loss sum，然后除以该样本有效 token 数，最后对 batch 求 mean
            #     loss = (valid_loss.sum(-1) / shift_mask.sum(-1).clamp(min=1.0)).mean()
            
            return loss

        loss_t1 = compute_masked_loss(
            logits_t1, labels_t1, loss_mask_t1,
            inputs["advantages_t1"], inputs["num_items_in_batch_t1"],
            inputs.get("old_per_token_logps_t1"), inputs.get("ref_per_token_logps_t1")
        )
        
        loss_t2 = compute_masked_loss(
            logits_t2, labels_t2, loss_mask_t2,
            inputs["advantages_t2"], inputs["num_items_in_batch_t2"],
            inputs.get("old_per_token_logps_t2"), inputs.get("ref_per_token_logps_t2")
        )


        # return (loss_t1 + loss_t2) / 2
        return loss_t2

    # def _compute_loss(self, model, inputs):
        
    #     # =================================================================================================
    #     # 内部函数：计算单轮 GRPO Loss
    #     # =================================================================================================
    #     def compute_single_turn_loss(
    #         prompt_ids, prompt_mask, completion_ids, completion_mask,
    #         advantages, num_items_in_batch,
    #         pixel_values=None, image_grid_thw=None, num_images=None, 
    #         pixel_attention_mask=None, image_sizes=None,
    #         old_per_token_logps=None, ref_per_token_logps=None
    #     ):
    #         # 1. 拼接 Input
    #         input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    #         attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
    #         logits_to_keep = completion_ids.size(1)
            
    #         # 2. 计算当前策略的 Logps
    #         # 注意：point_cloud 是共享的，直接从 inputs 取
    #         per_token_logps, entropies = self._get_per_token_logps_and_entropies(
    #             model,
    #             inputs["point_cloud"],
    #             input_ids,
    #             attention_mask,
    #             logits_to_keep,
    #             compute_entropy=True,
    #             pixel_values=pixel_values,
    #             image_grid_thw=image_grid_thw,
    #             num_images=num_images,
    #             pixel_attention_mask=pixel_attention_mask,
    #             image_sizes=image_sizes,
    #         )

    #         # 3. Entropy Regularization (Optional)
    #         if self.top_entropy_quantile < 1.0:
    #             entropy_mask = self.get_high_entropy_mask(entropies, completion_mask, 1 - self.top_entropy_quantile)
    #         else:
    #             entropy_mask = None

    #         # 4. KL Divergence (Optional)
    #         per_token_kl = torch.zeros_like(per_token_logps)
    #         if self.beta != 0.0 and ref_per_token_logps is not None:
    #             per_token_kl = (
    #                 torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
    #             )

    #         # 5. Importance Sampling Ratio
    #         if old_per_token_logps is None:
    #             old_per_token_logps = per_token_logps.detach()
            
    #         log_ratio = per_token_logps - old_per_token_logps
            
    #         if self.importance_sampling_level == "token":
    #             log_importance_weights = log_ratio
    #         elif self.importance_sampling_level == "sequence":
    #             log_importance_weights = (log_ratio * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
    #             log_importance_weights = log_importance_weights.unsqueeze(-1)
    #         else:
    #             raise ValueError("Unknown importance sampling level")

    #         coef_1 = torch.exp(log_importance_weights)
    #         coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
    #         if self.args.delta is not None:
    #             coef_1 = torch.clamp(coef_1, max=self.args.delta)

    #         # 6. Compute Clipped Loss
    #         per_token_loss1 = coef_1 * advantages.unsqueeze(1)
    #         per_token_loss2 = coef_2 * advantages.unsqueeze(1)
    #         per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
            
    #         if entropy_mask is not None:
    #             per_token_loss = per_token_loss * entropy_mask

    #         if self.use_vllm and self.vllm_importance_sampling_correction and "importance_sampling_ratio" in inputs:
    #             per_token_loss = per_token_loss * inputs["importance_sampling_ratio"]

    #         if self.beta != 0.0:
    #             per_token_loss = per_token_loss + self.beta * per_token_kl

    #         # 7. Aggregation & Normalization
    #         loss = 0.0
    #         if self.loss_type == "grpo":
    #             loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
    #         elif self.loss_type == "bnpo":
    #             loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
    #         elif self.loss_type == "dr_grpo":
    #             loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
    #         elif self.loss_type == "dapo":
    #             # DAPO Normalizer
    #             normalizer = num_items_in_batch / self.accelerator.num_processes
    #             loss = (per_token_loss * completion_mask).sum() / normalizer
    #         else:
    #             raise ValueError(f"Unknown loss type: {self.loss_type}")

    #         return loss / self.current_gradient_accumulation_steps

    #     # =================================================================================================
    #     # 主流程：分别计算两轮 Loss 并相加
    #     # =================================================================================================

    #     print("" * 20 + " DEBUG: Generation Shapes " + "*" * 20)
    #     print(f"[compute loss] pixel values shape: {inputs.get('pixel_values').shape}")
    #     print(f"=" * (40 + len(" DEBUG: Generation Shapes ") + 1))
        
    #     # --- 计算 Turn 1 Loss ---
    #     loss_t1 = compute_single_turn_loss(
    #         prompt_ids=inputs["prompt_ids_t1"],
    #         prompt_mask=inputs["prompt_mask_t1"],
    #         completion_ids=inputs["completion_ids_t1"],
    #         completion_mask=inputs["completion_mask_t1"],
    #         advantages=inputs["advantages_t1"],
    #         num_items_in_batch=inputs["num_items_in_batch_t1"],
    #         pixel_values=inputs.get("pixel_values"),
    #         image_grid_thw=inputs.get("image_grid_thw"),
    #         num_images=inputs.get("num_images"),
    #         pixel_attention_mask=inputs.get("pixel_attention_mask"),
    #         image_sizes=inputs.get("image_sizes"),
    #         old_per_token_logps=inputs.get("old_per_token_logps_t1"),
    #         ref_per_token_logps=inputs.get("ref_per_token_logps_t1")
    #     )
        
    #     # --- 计算 Turn 2 Loss ---
    #     loss_t2 = compute_single_turn_loss(
    #         prompt_ids=inputs["prompt_ids_t2"],
    #         prompt_mask=inputs["prompt_mask_t2"],
    #         completion_ids=inputs["completion_ids_t2"],
    #         completion_mask=inputs["completion_mask_t2"],
    #         advantages=inputs["advantages_t2"],
    #         num_items_in_batch=inputs["num_items_in_batch_t2"],
    #         pixel_values=inputs.get("pixel_values"),
    #         image_grid_thw=inputs.get("image_grid_thw"),
    #         num_images=inputs.get("num_images"),
    #         pixel_attention_mask=inputs.get("pixel_attention_mask"),
    #         image_sizes=inputs.get("image_sizes"),
    #         old_per_token_logps=inputs.get("old_per_token_logps_t2"),
    #         ref_per_token_logps=inputs.get("ref_per_token_logps_t2")
    #     )

    #     # --- 最终 Loss 求和 ---
    #     return loss_t1 + loss_t2

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: Optional[list[str]] = None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            loss = loss.mean().detach()
        return loss, None, None

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        super().log(logs, start_time)
        self._metrics[mode].clear()

        if self.accelerator.is_main_process and self.log_completions:
            if is_rich_available():
                print_prompt_completions_sample(
                    self._logs["prompt"],
                    self._logs["completion"],
                    self._logs["rewards"],
                    self._logs["advantages"],
                    self.state.global_step,
                    self.num_completions_to_print,
                )

            if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                import pandas as pd

                table = {
                    "step": [str(self.state.global_step)] * len(self._logs["prompt"]),
                    "prompt": self._logs["prompt"],
                    "completion": self._logs["completion"],
                    **self._logs["rewards"],
                    "advantage": self._logs["advantages"],
                }

                if self._logs["images"]:
                    table["images"] = []
                    for image_list in self._logs["images"]:
                        # Convert images to wandb Image objects for proper visualization
                        table["images"].append([wandb.Image(image) for image in image_list])

                df = pd.DataFrame(table)
                if self.wandb_log_unique_prompts:
                    df = df.drop_duplicates(subset=["prompt"])
                wandb.log({"completions": wandb.Table(dataframe=df)})

    # Ensure the model card is saved along with the checkpoint
    def _save_checkpoint(self, model, trial):
        if self.args.hub_model_id is None:
            model_name = Path(self.args.output_dir).name
        else:
            model_name = self.args.hub_model_id.split("/")[-1]
        self.create_model_card(model_name=model_name)
        super()._save_checkpoint(model, trial)

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str`, *optional*):
                Name of the model.
            dataset_name (`str`, *optional*):
                Name of the dataset used for training.
            tags (`str`, `list[str]`, *optional*):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        # normalize `tags` to a mutable set
        if tags is None:
            tags = set()
        elif isinstance(tags, str):
            tags = {tags}
        else:
            tags = set(tags)

        if hasattr(self.model.config, "unsloth_version"):
            tags.add("unsloth")

        if "JOB_ID" in os.environ:
            tags.add("hf_jobs")

        tags.update(self._tag_names)

        # docstyle-ignore
        citation = textwrap.dedent(
            """\
            @article{shao2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            }
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=list(tags),
            wandb_url=wandb.run.url if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
