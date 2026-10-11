# PointVL-3D

**Reinforcing Structured 3D Grounding via Geometric Reward Credit Assignment**

Jingkun Chen, Ruoshi Xu, Mingqi Gao, Shengda Luo, and Jungong Han

PointVL-3D studies geometry-aware reinforcement learning for point–vision–language models that generate text, 2D/3D bounding boxes, and keypoints in a single structured response. The paper introduces **Geometric Reward Credit Assignment (GRCA)** to route field-specific rewards to the responsible output tokens and **Reprojection Consistency (RPC)** to measure agreement between predicted 2D and 3D boxes.

This repository contains the original research source snapshot. The method and results below summarize the accompanying manuscript; the source has been uploaded without modifying its training or evaluation logic. See [Implementation status](#implementation-status) for differences between the supplied snapshot and the manuscript.

## Method

The point–image policy combines **Qwen2.5-VL-3B-Instruct** with **Point-BERT** and generates a JSON response with two semantic fields and four geometric fields.

1. **Structural warm-up:** supervised fine-tuning teaches the model the structured output format. The manuscript describes continuous coordinates quantized into 1001 bins.
2. **GRCA:** each geometric field is evaluated separately. Its group-relative advantage is assigned to the tokens that produce that field, while other tokens receive a background advantage.
3. **RPC:** the eight corners of the predicted axis-aligned 3D box are projected using calibrated camera parameters. RPC is the IoU between their enclosing image box and the model's predicted 2D box.

For field $f$, rewards are standardized within the $G$ responses sampled for the same input:

$$
\widetilde A_f = \frac{r_f - \mu_f}{\sigma_f + \epsilon}.
$$

The token-wise advantage is

$$
\mathcal A_t = \sum_{f\in\mathcal F} m_{t,f}\widetilde A_f + m_{t,\mathrm{bg}}\widetilde A_{\mathrm{bg}},
$$

where $\mathcal F=\{\texttt{bbox2d},\texttt{bbox3d},\texttt{kpts2d},\texttt{kpts3d}\}$. RPC enters the background-token advantage:

$$
\widetilde A_{\mathrm{bg}} = \frac{1-\lambda}{4}\sum_{f\in\mathcal F}\widetilde A_f + \lambda\widetilde A_{\mathrm{rpc}}.
$$

The manuscript uses $\lambda=0.15$ for the full model. GRCA retains the clipped GRPO objective and adds no inference-time credit-assignment module.

## Structured outputs

The manuscript and the supplied source use different names for the same geometric fields:

| Manuscript field | Source field | Meaning |
| --- | --- | --- |
| `answer` | `answer` | Semantic answer |
| `description` | `description` | Object description |
| `bbox2d` | `bbox` | 2D box: `[xmin, ymin, xmax, ymax]` |
| `bbox3d` | `3d_bbox` | Axis-aligned 3D box: `[xmin, ymin, zmin, xmax, ymax, zmax]` |
| `kpts2d` | `point_2d` | List of 2D points |
| `kpts3d` | `point_3d` | List of 3D points |

The geometric reward functions also accept a 3D box represented by two corner lists. **KPA is containment-based:** it measures the fraction of generated points inside the corresponding ground-truth box, rather than point-to-point semantic correspondence.

## Results reported in the manuscript

These values are copied from the manuscript. They have not been independently reproduced as part of this source upload.

### In-domain post-training ablation

| Method | IoU 2D ↑ | IoU 3D ↑ | KPA 2D ↑ | KPA 3D ↑ | RPC ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| SFT | 0.890 | 0.663 | 0.890 | 0.640 | 0.836 |
| SFT, budget-matched | 0.888 | 0.678 | 0.880 | 0.670 | 0.844 |
| GRPO, broadcast | 0.886 | 0.684 | 0.890 | 0.890 | 0.841 |
| GRCA | 0.894 | 0.683 | 0.890 | 0.930 | 0.846 |
| **GRCA + RPC** | **0.894** | **0.686** | **0.890** | **0.930** | **0.852** |

Under the same group-relative optimization framework, GRCA improves KPA 3D from 0.890 to 0.930. RPC measures agreement between predictions and should be interpreted together with ground-truth box IoU.

### Direct transfer to NOCS

| Model | IoU 2D ↑ | IoU 3D ↑ |
| --- | ---: | ---: |
| PointVL-3D | 0.5342 | 0.4263 |

No NOCS samples are used for fine-tuning. The manuscript does not report NOCS KPA or RPC.

### Language evaluation

| Model | BLEU-4 ↑ | ROUGE-L ↑ | BERTScore ↑ | QA accuracy ↑ |
| --- | ---: | ---: | ---: | ---: |
| SFT | 0.1819 | 0.4093 | 0.9154 | 0.953 |
| GRCA | 0.1767 | 0.4075 | 0.9152 | 0.951 |
| GRCA + RPC | 0.1823 | 0.4122 | 0.9154 | 0.951 |

## Dataset and training setup

The manuscript evaluates a **Generative-Refined 3D-VL benchmark derived from ShapeNetCore**, with 12,347 instances across 51 categories: 11,347 training samples and 1,000 test samples. Each instance combines a 1024-point cloud, an RGB image, language annotations, and structured geometric targets. Reprojection-based experiments use camera-calibrated samples retained after manual screening for geometric drift.

| Setting | Stage 1: SFT | Stage 2: RL |
| --- | --- | --- |
| Epochs | 2 | 1 |
| Learning rate | $2\times10^{-5}$ to $2\times10^{-7}$, cosine schedule | $1\times10^{-5}$ |
| Batch / group size | Batch size 128 | Group size 8 |
| Image resolution | 512 × 512 | 512 × 512 |

The manuscript reports experiments on four NVIDIA A800 80 GB GPUs. Dataset files and trained checkpoints are not included in this source snapshot.

## Repository layout

```text
model_arch/       Point–image Qwen2.5-VL model implementations
trl/scripts/      Supervised and reinforcement-learning entry points
trl/trainer/      Customized trainers and supporting TRL source
trl/rewards/      Geometric rewards and field-span reward assignment
commands/         Original experiment launch scripts
configs/          Distributed training configuration
eval/             Grounding, language, and visualization evaluation scripts
```

Key files:

- [Point–image model](model_arch/modeling_qwen2_5_pvl_batch.py)
- [SFT entry point](trl/scripts/sft_pvlm.py) and [trainer](trl/trainer/sft_pllm_trainer.py)
- [Field-routed RL entry point](trl/scripts/grpo_pllm_dr.py) and [trainer](trl/trainer/grpo_pllm_dr_trainer.py)
- [Field-routed geometric rewards](trl/rewards/pvl_dr.py)
- [Geometry evaluation](eval/eval_geo.py)
- [Language and QA evaluation](eval/eval_qa_nlp.py)

## Using the source snapshot

Clone the repository:

```bash
git clone https://github.com/jingkunchen/PointVL-3D.git
cd PointVL-3D
```

The scripts depend on PyTorch, Transformers, Accelerate, Datasets, Safetensors, NumPy, Pillow, and the corresponding Point-BERT implementation. Additional dependencies vary by evaluation script. A verified environment lockfile is not included.

Before running, prepare the dataset and checkpoints, restore the missing upstream dependencies described below, and adapt the original server paths and GPU settings to your environment. The existing launch scripts are experiment references:

```bash
# Stage 1: structural supervised fine-tuning
bash commands/run_sft_vlm.sh

# Stage 2: field-routed reinforcement learning
bash commands/run_grpo_dr.sh
```

The dataset readers use `messages`, `images` or `image`, and `point_cloud`. The reward path reads structured ground truth from the assistant message and camera parameters from `ext_param`, including `K`, `R`, `T`, and `image_size`. The supplied projection helper uses the PyTorch3D row-vector convention `points @ R + T`, followed by the x/y sign conversion to OpenCV coordinates. Camera conventions and coordinate units must match the data.

## Implementation status

This upload preserves the supplied code. It is a research snapshot, with the following reproducibility details:

- The original launch scripts contain absolute server paths and experiment-specific settings; their values differ from some manuscript settings.
- Point-BERT source/configuration and weights referenced by the model are not included. Some upstream TRL modules, including `trl/data_utils.py` and `trl/models/`, are also absent from the supplied snapshot and must be restored from the matching upstream environment before execution.
- In `pvl_dr.py`, the current `geo` helper measures coverage of the **ground-truth** 2D box by the projected 3D box. The manuscript defines RPC as IoU with the **predicted** 2D box. The supplied code also hard-codes a 0.125 mixing weight.
- The supplied dense-reward function normalizes over its local batch. Its assumption that the batch forms one rollout group must be checked when using multiple prompt groups or distributed rollouts.

The uploaded snapshot should therefore not be treated as a verified, exact reproduction of the manuscript's full GRCA + RPC configuration.

## Acknowledgments and license

The implementation builds on [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL), [Point-BERT](https://github.com/Julie-tang00/Point-BERT), and [Hugging Face TRL](https://github.com/huggingface/trl). Existing upstream copyright notices are retained. This repository includes an [Apache 2.0 license](LICENSE); third-party components remain subject to their respective licenses.
