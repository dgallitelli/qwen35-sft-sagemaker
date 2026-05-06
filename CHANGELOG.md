# Changelog

All notable user-facing changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions are not strictly tagged — entries are grouped by merge date.

## Unreleased

### Documented
- README "Default Hyperparameters" section listing every recipe setting that
  the validation matrix was run against, plus a memory-footprint hint for 9B
  full SFT (~43 GB peak per GPU).
- 9B Base full SFT on `ml.g6e.12xlarge` (4×L40S 192 GB total) marked **Does
  not fit (OOM)**. The recipe defaults peak at ~43 GB per GPU during the
  backward pass, exceeding L40S usable VRAM (~44.4 GB after CUDA overhead).
  Use `ml.g7e.12xlarge` (4×96 GB Blackwell) or tune the recipe down
  (smaller batch / seq length / optimizer offload).

### Changed (previously)
- `launch_sft_job.py`: removed the silent `DEFAULT_REGION_FALLBACK` (`ap-southeast-1`).
  When no region is configured (no `AWS_REGION` / `AWS_DEFAULT_REGION` env var,
  no profile, no IMDS), the script now raises with a clear message instead of
  shipping training jobs to a surprise region. Pass `--region` or set
  `AWS_REGION` to control destination explicitly.

## 2026-05-05

### Added
- Recipes for the Qwen3.5 post-trained ("Instruct") variants:
  `Qwen3.5-{4B,9B}--vanilla-{peft-qlora,full}.yaml`. On HuggingFace these
  models are published as `Qwen/Qwen3.5-{4B,9B}` with **no `-Instruct`
  suffix** — the `-Base` suffix denotes the pretrained checkpoint.
- `launch_sft_job.py` flags: `--variant {base,instruct}`,
  `--instance-type` override, `--base-job-name` override.
- Open-source recipe-validation harness under `experiments/`:
  `matrix.template.json` (account-agnostic test matrix),
  `launch_matrix.py` (submits each pending row),
  `monitor_matrix.py` (polls until terminal status), and `README.md`.
- Smoke-test dataset at `data/sft-dataset.jsonl` (100 synthetic Q/A pairs).

### Changed
- Full-FT defaults moved from the untested `ml.p4d.24xlarge` to validated
  single-node options: `ml.g7e.2xlarge` for 4B and `ml.g7e.12xlarge` for 9B.
- README "Instance Recommendations" table updated accordingly; new
  validation matrix added documenting all eight recipe×instance combinations
  end-to-end on real SageMaker training jobs.
