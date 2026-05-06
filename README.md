# Qwen 3.5 SFT Recipes — Qwen 3.5 on SageMaker

Validated SFT recipes for fine-tuning **Qwen 3.5** (4B and 9B, Base and post-trained) on Amazon SageMaker Training Jobs using the [SageMaker Generative AI Recipes](https://github.com/aws-samples/amazon-sagemaker-generativeai) framework.

## What's Included

> **Variant naming:** On HuggingFace, the post-trained ("Instruct") models are published as `Qwen/Qwen3.5-{4B,9B}` with **no `-Instruct` suffix** — the `-Base` suffix denotes the pretrained checkpoint. Both variants share the same `Qwen3_5ForConditionalGeneration` architecture, so the same DLC and dependency pins apply; only weights and `chat_template.jinja` differ. The launcher exposes this with `--variant {base,instruct}` (default: `base`).

| Recipe | Model | Strategy | Tested Instance | Status |
|--------|-------|----------|----------------|--------|
| `Qwen3.5-4B-Base--vanilla-peft-qlora.yaml` | Qwen/Qwen3.5-4B-Base | QLoRA (4-bit) | ml.g5.2xlarge | Validated |
| `Qwen3.5-9B-Base--vanilla-peft-qlora.yaml` | Qwen/Qwen3.5-9B-Base | QLoRA (4-bit) | ml.g5.2xlarge | Validated |
| `Qwen3.5-4B-Base--vanilla-full.yaml` | Qwen/Qwen3.5-4B-Base | Full fine-tuning | ml.g7e.2xlarge | Validated |
| `Qwen3.5-9B-Base--vanilla-full.yaml` | Qwen/Qwen3.5-9B-Base | Full fine-tuning | ml.g7e.12xlarge | Validated |
| `Qwen3.5-9B-Base--vanilla-full.yaml` | Qwen/Qwen3.5-9B-Base | Full fine-tuning | ml.g6e.12xlarge | Does not fit (OOM) — see [Default Hyperparameters](#default-hyperparameters) |
| `Qwen3.5-4B--vanilla-peft-qlora.yaml` | Qwen/Qwen3.5-4B (Instruct) | QLoRA (4-bit) | ml.g5.2xlarge | Validated |
| `Qwen3.5-9B--vanilla-peft-qlora.yaml` | Qwen/Qwen3.5-9B (Instruct) | QLoRA (4-bit) | ml.g5.2xlarge | Validated |
| `Qwen3.5-9B--vanilla-peft-qlora.yaml` | Qwen/Qwen3.5-9B (Instruct) | QLoRA (4-bit) | ml.g6e.2xlarge | Validated |
| `Qwen3.5-4B--vanilla-full.yaml` | Qwen/Qwen3.5-4B (Instruct) | Full fine-tuning | ml.g7e.2xlarge | Validated |
| `Qwen3.5-9B--vanilla-full.yaml` | Qwen/Qwen3.5-9B (Instruct) | Full fine-tuning | ml.g7e.12xlarge | Validated |

**QLoRA test results** (Qwen3.5-9B-Base, 900 samples from AI-MO/NuminaMath-CoT, 1 epoch):

| Instance | GPU | vRAM | Wall Clock | Billable |
|----------|-----|------|------------|----------|
| ml.g5.2xlarge | A10G | 24 GB | ~40 min | ~40 min |
| ml.g6.4xlarge | L4 | 24 GB | ~35.5 min | ~35.5 min |
| ml.g7e.2xlarge | RTX PRO 6000 Blackwell | 96 GB | ~19 min | ~19 min |

No recipe changes were required when switching between instance types within the same family — the same YAML works across all tested instances.

**Instruct-variant smoke-test results** (100 synthetic Q/A pairs, 10 epochs, default recipe hyperparameters):

| # | Variant | Strategy | Instance | GPU(s) | Billable |
|---|---------|----------|----------|--------|----------|
| T1 | 4B Instruct | QLoRA | ml.g5.2xlarge | 1× A10G 24 GB | ~21 min |
| T2 | 4B Base | Full SFT | ml.g7e.2xlarge | 1× RTX PRO 6000 Blackwell 96 GB | ~29 min |
| T3 | 4B Instruct | Full SFT | ml.g7e.2xlarge | 1× RTX PRO 6000 Blackwell 96 GB | ~30 min |
| T4 | 9B Base | Full SFT | ml.g7e.12xlarge | 4× RTX PRO 6000 Blackwell 384 GB total | ~49 min |
| T5 | 9B Instruct | Full SFT | ml.g7e.12xlarge | 4× RTX PRO 6000 Blackwell 384 GB total | ~46 min |
| T6 | 9B Instruct | QLoRA | ml.g5.2xlarge | 1× A10G 24 GB | ~28 min |
| T7 | 9B Instruct | QLoRA | ml.g6e.2xlarge | 1× L40S 48 GB | ~22 min |

The Instruct recipes pass the same `model_name_or_path` (drop the `-Base` suffix) and otherwise inherit the Base recipe wholesale — no DLC, dependency, or trainer changes were needed.

## Default Hyperparameters

Every "Validated" cell above ran with the recipe defaults below, on 100 synthetic Q/A pairs for 10 epochs. **Recipe defaults are tuned to fit on the recommended instance — if you change them, you may need a larger instance.**

| Setting | Value |
|---------|-------|
| `torch_dtype` | `bfloat16` |
| `attn_implementation` | `sdpa` |
| `bf16` | `true` |
| `tf32` | `false` |
| `use_liger` | `false` |
| `max_seq_length` | `4096` |
| `packing` | `false` |
| `modality_type` | `"text"` |
| `per_device_train_batch_size` | `2` |
| `gradient_accumulation_steps` | `2` |
| `gradient_checkpointing` | `true` (with `use_reentrant: true`) |
| `num_train_epochs` | `10` |
| `learning_rate` | `1.0e-4` |
| `lr_scheduler_type` | `cosine` |
| `warmup_ratio` | `0.1` |
| `seed` | `42` |
| **QLoRA-only** | |
| `load_in_4bit` | `true` |
| `lora_target_modules` | `["q_proj", "k_proj", "v_proj", "o_proj"]` |
| `lora_r` / `lora_alpha` | `8` / `16` |

**Memory footprint hint** (9B full SFT, default recipe): ZeRO-3 with `per_device_train_batch_size=2` and `max_seq_length=4096` consumes ~43 GB per GPU at peak — fits comfortably on `ml.g7e.12xlarge` (4× 96 GB), **does not fit** on `ml.g6e.12xlarge` (4× 48 GB) and OOMs in the backward pass. To fit on smaller GPUs, lower `per_device_train_batch_size` to 1 (with `gradient_accumulation_steps=4` to keep the effective batch size), reduce `max_seq_length`, and/or enable optimizer offload to CPU in `sagemaker_code/configs/accelerate/ds_zero3.yaml`.

## Quick Start

### 1. Prepare your dataset

Your dataset should be a JSONL file in chat messages format:

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Place it at `data/sft-dataset.jsonl` (or pass `--dataset-s3` / `--dataset-local`).

### 2. Launch a training job

```bash
pip install sagemaker boto3

# QLoRA on 9B Base (default) — runs on ml.g5.2xlarge
python launch_sft_job.py

# QLoRA on 4B Base
python launch_sft_job.py --model 4b

# QLoRA on 4B Instruct (post-trained variant)
python launch_sft_job.py --variant instruct --model 4b

# Full fine-tuning on 9B Base — runs on ml.g7e.12xlarge by default
python launch_sft_job.py --model 9b --strategy full

# Full fine-tuning on 9B Instruct — runs on ml.g7e.12xlarge by default
python launch_sft_job.py --variant instruct --model 9b --strategy full

# Point to dataset already in S3
python launch_sft_job.py --dataset-s3 s3://my-bucket/data/sft-dataset.jsonl

# Override the instance type chosen by the (variant, model, strategy) mapping
python launch_sft_job.py --model 4b --strategy full --instance-type ml.g6e.4xlarge

# Sideload a SageMaker inference handler so it ships inside model.tar.gz
# at code/inference.py — picked up automatically by the HF Inference DLC.
# No post-job download/untar/repack/re-upload dance required.
python launch_sft_job.py \
    --inference-handler ./inference.py \
    --inference-requirements ./requirements.txt   # optional
```

### 3. Customize the recipe

Edit the YAML files in `sagemaker_code/hf_recipes/Qwen/` to change:
- `num_train_epochs` — currently set to 10
- `learning_rate` — default 1e-4
- `max_seq_length` — default 4096
- `lora_r` / `lora_alpha` — for QLoRA recipes
- `report_to` — currently `mlflow`, can also use `tensorboard`

Or generate new recipes with the interactive tool:
```bash
python sft_recipe_generator.py --easy
```

## DLC and Dependency Notes

These recipes are validated against the **PyTorch 2.9.0 DLC** (`pytorch-training:2.9.0-gpu-py312-cu130-ubuntu22.04-sagemaker`).

Qwen 3.5 uses the `qwen3_5` model architecture, which required several dependency bumps from the DLC defaults:

| Package | DLC Default | Required | Why |
|---------|------------|----------|-----|
| transformers | 4.x | 5.2.0 | `qwen3_5` architecture not in 4.x |
| peft | 0.17.0 | 0.18.1 | HybridCache removed in transformers 5.x |
| bitsandbytes | 0.46.1 | 0.49.2 | No CUDA 13.0 binary in 0.46.x |
| liger-kernel | 0.6.1 | 0.7.0 | Same HybridCache compatibility issue |

All of these are pinned in `sagemaker_code/requirements.txt`.

**Attention implementation**: `sdpa` (scaled dot-product attention) is used instead of `flash_attention_2`, which has import issues with transformers 5.x on this DLC.

## Qwen 3.5 — Multimodal Note

Qwen 3.5 is natively multimodal (vision-language). For text-only SFT, set `modality_type: "text"` in the recipe (already done in all included recipes). The text-only training path works identically to Qwen 3.

## Instance Recommendations

| Strategy | 4B | 9B |
|----------|-----|-----|
| QLoRA | ml.g5.2xlarge (1× A10G, 24 GB) | ml.g5.2xlarge (1× A10G, 24 GB) |
| Full fine-tuning | ml.g7e.2xlarge (1× RTX PRO 6000 Blackwell, 96 GB) | ml.g7e.12xlarge (4× RTX PRO 6000 Blackwell, 384 GB total) |

> V100 instances (p3 family) do **not** support bf16, which is required for Qwen3/3.5. Use g5 or newer.

These defaults are wired into `launch_sft_job.py` via the `(variant, model, strategy)` mapping; pass `--instance-type` to override.

## Reproducing the validation matrix

The `experiments/` folder contains the harness used to validate every recipe
in this repo against real SageMaker training jobs:

```bash
# 1. Set the SageMaker execution role you want training jobs to assume
export SAGEMAKER_ROLE_ARN="arn:aws:iam::<account>:role/<role-name>"

# 2. Submit every pending row from the matrix
python experiments/launch_matrix.py

# 3. (Optional) poll status until all jobs reach a terminal state
python experiments/monitor_matrix.py
```

`experiments/matrix.template.json` is the public, account-agnostic test
matrix. The first run bootstraps from it into `experiments/results.json`
(gitignored — contains your account-specific S3 URI and training-job names).
See `experiments/README.md` for details.

## Repo Structure

```
├── launch_sft_job.py                    # Training job launcher (SDK v3)
├── sft_recipe_generator.py              # Interactive recipe generator
├── data/sft-dataset.jsonl               # Smoke-test dataset (100 synthetic Q/A pairs)
├── experiments/                         # Recipe validation harness
│   ├── matrix.template.json             # Public, account-agnostic test matrix
│   ├── launch_matrix.py                 # Submits each pending row
│   └── monitor_matrix.py                # Polls until terminal status
└── sagemaker_code/
    ├── sft.py                           # Training entrypoint
    ├── sm_accelerate_train.sh           # Accelerate launch wrapper
    ├── requirements.txt                 # Pinned dependencies
    ├── inference.py                     # Model serving entrypoint
    ├── configs/                         # Accelerate / DeepSpeed configs
    ├── utils/                           # Helpers (FLOPs meter, adapter merge)
    └── hf_recipes/Qwen/                 # Training recipe YAMLs
        ├── Qwen3.5-4B-Base--vanilla-peft-qlora.yaml
        ├── Qwen3.5-9B-Base--vanilla-peft-qlora.yaml
        ├── Qwen3.5-4B-Base--vanilla-full.yaml
        ├── Qwen3.5-9B-Base--vanilla-full.yaml
        ├── Qwen3.5-4B--vanilla-peft-qlora.yaml         # Instruct variant
        ├── Qwen3.5-9B--vanilla-peft-qlora.yaml         # Instruct variant
        ├── Qwen3.5-4B--vanilla-full.yaml               # Instruct variant
        └── Qwen3.5-9B--vanilla-full.yaml               # Instruct variant
```

## Changelog

User-facing changes are tracked in [CHANGELOG.md](CHANGELOG.md).

## Credits

Training infrastructure from [amazon-sagemaker-generativeai](https://github.com/aws-samples/amazon-sagemaker-generativeai). Recipes and dependency fixes for Qwen 3.5 by AWS.
