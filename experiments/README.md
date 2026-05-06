# Experiments — recipe validation harness

This folder contains the test harness used to validate the recipes against
real SageMaker training jobs.

## Files

| File | Purpose |
|------|---------|
| `matrix.template.json` | Canonical, account-agnostic test matrix (committed). |
| `launch_matrix.py` | Submits each `pending` row to SageMaker via `launch_sft_job.py`. |
| `monitor_matrix.py` | Polls running jobs every 60s and updates terminal status. |
| `results.json` | Per-run state (training-job names, billable seconds). **Gitignored** — recreated on each run by bootstrapping from the template. |

## Usage

```bash
# 1. Make sure launch_sft_job.py works for you (see top-level README).
# 2. Set your SageMaker execution role:
export SAGEMAKER_ROLE_ARN="arn:aws:iam::<account>:role/<role-name>"

# 3. Submit every pending row in the matrix.
python experiments/launch_matrix.py

# 4. (Optional) poll until everything finishes.
python experiments/monitor_matrix.py
```

The launcher uploads `data/sft-dataset.jsonl` to your default SageMaker
bucket and reuses the same S3 URI for every run in the matrix. Each row's
training-job name and final status get persisted back to `results.json` so
the script is idempotent — re-running it only resubmits rows that failed
to submit and skips ones already running or complete.
