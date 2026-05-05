"""
Submit the test matrix from results.json. For each test row whose
submit_status == 'pending', call launch_sft_job.main(...) by composing
argv, then capture the resulting TrainingJobName + S3 paths and patch
results.json in place.

Idempotent: rows already 'submitted' or further are skipped.
"""
import json
import os
import sys
import time
import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = REPO_ROOT / "experiments" / "results.json"
TEMPLATE_PATH = REPO_ROOT / "experiments" / "matrix.template.json"
sys.path.insert(0, str(REPO_ROOT))

import launch_sft_job


def now_iso():
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def submit_row(row, dataset_s3, region):
    """
    Compose argv, run launch_sft_job.main, and capture the TrainingJobName
    via a monkey-patch on ModelTrainer.train.
    """
    # SageMaker TrainingJobName regex disallows '.', so collapse 'T4.1' -> 't4-1'.
    safe_id = row["id"].lower().replace(".", "-")
    job_tag = (
        f"qwen35-{row['model']}-{row['variant']}-{row['strategy']}-{safe_id}"
    )

    # Force a deterministic base_job_name so we can trace runs by test id.
    role_arn = os.environ.get("SAGEMAKER_ROLE_ARN")
    sys.argv = [
        "launch_sft_job.py",
        "--variant", row["variant"],
        "--model", row["model"],
        "--strategy", row["strategy"],
        "--instance-type", row["instance_type"],
        "--region", region,
        "--dataset-s3", dataset_s3,
        "--base-job-name", job_tag,
    ]
    if role_arn:
        sys.argv.extend(["--role", role_arn])
    print(f"\n{'='*72}\n[{row['id']}] launching: {' '.join(sys.argv[1:])}\n{'='*72}", flush=True)

    # Capture state from the trainer before train() returns. ModelTrainer
    # populates _latest_training_job synchronously inside train() when
    # mode is SAGEMAKER_TRAINING_JOB and the API call succeeds, so we
    # don't need to monkey-patch — launch_sft_job already prints the
    # TrainingJobName. But we re-run resolution here from the trainer
    # object via a small hook.
    captured = {}
    orig_train = launch_sft_job.ModelTrainer.train
    def patched_train(self, *a, **kw):
        result = orig_train(self, *a, **kw)
        captured["trainer"] = self
        return result
    launch_sft_job.ModelTrainer.train = patched_train
    try:
        launch_sft_job.main()
    finally:
        launch_sft_job.ModelTrainer.train = orig_train

    trainer = captured.get("trainer")
    job_name = None
    if trainer is not None:
        latest = getattr(trainer, "_latest_training_job", None)
        job_name = getattr(latest, "training_job_name", None)
    return job_name


def main():
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            results = json.load(f)
    else:
        # First run: bootstrap from the public template (no account ids).
        with open(TEMPLATE_PATH) as f:
            results = json.load(f)
    region = results.get("region") or "us-east-1"

    # Upload dataset once (or use --dataset-s3 if already uploaded).
    dataset_s3 = results.get("dataset_s3")
    if not dataset_s3:
        import boto3
        from sagemaker.core.helper.session_helper import Session
        sess = Session(boto_session=boto3.Session(region_name=region))
        bucket = sess.default_bucket()
        local_path = REPO_ROOT / "data" / "sft-dataset.jsonl"
        s3_key = "qwen35-sft/dataset/sft-dataset.jsonl"
        s3 = boto3.client("s3", region_name=region)
        print(f"Uploading {local_path} -> s3://{bucket}/{s3_key}", flush=True)
        s3.upload_file(str(local_path), bucket, s3_key)
        dataset_s3 = f"s3://{bucket}/{s3_key}"
        results["dataset_s3"] = dataset_s3
        results["started_at_utc"] = now_iso()
        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2)

    for row in results["tests"]:
        if row.get("submit_status") not in (None, "pending", "failed_to_submit"):
            print(f"[{row['id']}] already in state '{row['submit_status']}' — skipping", flush=True)
            continue
        try:
            job_name = submit_row(row, dataset_s3, region)
            row["training_job_name"] = job_name
            row["submit_status"] = "submitted" if job_name else "submitted_no_name_captured"
            row["submitted_at_utc"] = now_iso()
        except Exception as e:
            row["submit_status"] = "failed_to_submit"
            row["submit_error"] = repr(e)
            print(f"[{row['id']}] submit failed: {e!r}", flush=True)
        # Persist after every row so a partial failure doesn't lose state.
        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2)

    # Summary
    print("\n=== submission summary ===")
    for row in results["tests"]:
        print(f"  {row['id']}: {row['submit_status']:25s}  job={row.get('training_job_name')}")


if __name__ == "__main__":
    main()
