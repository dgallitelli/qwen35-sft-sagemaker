# Qwen 3.5 SFT Recipes — Qwen 3.5 on SageMaker

Validated SFT recipes for fine-tuning **Qwen 3.5** (4B and 9B Base) on Amazon SageMaker Training Jobs using the [SageMaker Generative AI Recipes](https://github.com/aws-samples/amazon-sagemaker-generativeai) framework.

## What's Included

| Recipe | Model | Strategy | Tested Instance | Status |
|--------|-------|----------|----------------|--------|
| `Qwen3.5-4B-Base--vanilla-peft-qlora.yaml` | Qwen/Qwen3.5-4B-Base | QLoRA (4-bit) | ml.g5.2xlarge | Validated |
| `Qwen3.5-9B-Base--vanilla-peft-qlora.yaml` | Qwen/Qwen3.5-9B-Base | QLoRA (4-bit) | ml.g5.2xlarge | Validated |
| `Qwen3.5-4B-Base--vanilla-full.yaml` | Qwen/Qwen3.5-4B-Base | Full fine-tuning | ml.p4d.24xlarge | Not yet tested |
| `Qwen3.5-9B-Base--vanilla-full.yaml` | Qwen/Qwen3.5-9B-Base | Full fine-tuning | ml.p4d.24xlarge | Not yet tested |

**QLoRA test results** (900 samples from AI-MO/NuminaMath-CoT, 1 epoch, ml.g5.2xlarge):
- 4B: train_loss=0.536, ~30 min
- 9B: train_loss=0.508, ~40 min

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

# QLoRA on 9B (default) — runs on ml.g5.2xlarge
python launch_sft_job.py

# QLoRA on 4B
python launch_sft_job.py --model 4b

# Full fine-tuning on 9B — requires ml.p4d.24xlarge
python launch_sft_job.py --model 9b --strategy full

# Point to dataset already in S3
python launch_sft_job.py --dataset-s3 s3://my-bucket/data/sft-dataset.jsonl
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
| QLoRA | ml.g5.2xlarge (1x A10G, 24GB) | ml.g5.2xlarge (1x A10G, 24GB) |
| Full fine-tuning | ml.g5.12xlarge (4x A10G, 96GB) | ml.p4d.24xlarge (8x A100, 320GB) |

> V100 instances (p3 family) do **not** support bf16, which is required for Qwen3/3.5. Use g5 or newer.

## Repo Structure

```
├── launch_sft_job.py                    # Training job launcher (SDK v3)
├── sft_recipe_generator.py              # Interactive recipe generator
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
        └── Qwen3.5-9B-Base--vanilla-full.yaml
```

## Credits

Training infrastructure from [amazon-sagemaker-generativeai](https://github.com/aws-samples/amazon-sagemaker-generativeai). Recipes and dependency fixes for Qwen 3.5 by AWS.
