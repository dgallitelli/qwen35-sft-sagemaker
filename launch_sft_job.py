"""
Launch SageMaker Training Job for Qwen3.5 SFT
Uploads a local JSONL dataset to S3, then starts a training job.
SageMaker Python SDK v3 (ModelTrainer)

Usage:
    python launch_sft_job.py                                    # defaults: 9B QLoRA
    python launch_sft_job.py --model 4b --strategy qlora        # 4B QLoRA
    python launch_sft_job.py --model 9b --strategy full         # 9B full fine-tuning

    # Sideload a SageMaker inference handler so it lands in model.tar.gz
    # at code/inference.py — no manual repacking needed.
    python launch_sft_job.py --inference-handler ./inference.py \
        [--inference-requirements ./requirements.txt]

Prerequisites:
    pip install sagemaker boto3 datasets
"""
import argparse
import os
import shutil
import tempfile
import boto3
from sagemaker.core.helper.session_helper import Session, get_execution_role
from sagemaker.train.model_trainer import ModelTrainer
from sagemaker.train.configs import (
    InputData,
    Compute,
    SourceCode,
    OutputDataConfig,
    StoppingCondition,
    CheckpointConfig,
)

# --- Config (update these for your account) ---
# Fallback region used only when the caller's AWS environment has no
# region configured (no AWS_REGION / AWS_DEFAULT_REGION env var, no
# ~/.aws/config profile region, no --region flag). See resolve_region().
DEFAULT_REGION_FALLBACK = "ap-southeast-1"
ROLE_ARN = None  # Set to your SageMaker execution role ARN, or None to auto-detect
DATASET_S3_URI = None  # Set to your S3 dataset URI, or None to upload LOCAL_DATASET_PATH
LOCAL_DATASET_PATH = "data/sft-dataset.jsonl"  # Local path to upload if DATASET_S3_URI is None
S3_PREFIX = "qwen35-sft"  # S3 prefix for outputs and dataset uploads

# DLC image — PyTorch 2.9.0, CUDA 13.0, Python 3.12
DLC_TAG = "2.9.0-gpu-py312-cu130-ubuntu22.04-sagemaker"

# Recipe and instance mapping
RECIPES = {
    ("4b", "qlora"): ("hf_recipes/Qwen/Qwen3.5-4B-Base--vanilla-peft-qlora.yaml", "ml.g5.2xlarge"),
    ("9b", "qlora"): ("hf_recipes/Qwen/Qwen3.5-9B-Base--vanilla-peft-qlora.yaml", "ml.g5.2xlarge"),
    ("4b", "full"):  ("hf_recipes/Qwen/Qwen3.5-4B-Base--vanilla-full.yaml", "ml.p4d.24xlarge"),
    ("9b", "full"):  ("hf_recipes/Qwen/Qwen3.5-9B-Base--vanilla-full.yaml", "ml.p4d.24xlarge"),
}


def resolve_region():
    """Pick a region in this order:

    1. boto3 default (botocore's resolution chain: ``AWS_REGION`` then
       ``AWS_DEFAULT_REGION`` env vars, then the active ``~/.aws/config``
       profile, then the EC2 IMDS region endpoint unless
       ``AWS_EC2_METADATA_DISABLED=true``), so the launcher matches the
       user's existing AWS setup without surprise.
    2. ``DEFAULT_REGION_FALLBACK`` if nothing is configured. A warning
       is printed when the fallback fires so callers notice their AWS
       env has no region set rather than silently shipping to it.

    The ``--region`` CLI flag still overrides this resolution.
    """
    region = boto3.Session().region_name
    if region:
        return region
    print(
        f"WARNING: no AWS region resolved from env / profile / IMDS; "
        f"falling back to {DEFAULT_REGION_FALLBACK}. Set AWS_REGION or "
        f"pass --region to silence this."
    )
    return DEFAULT_REGION_FALLBACK


def parse_args():
    parser = argparse.ArgumentParser(description="Launch Qwen3.5 SFT on SageMaker")
    parser.add_argument("--model", choices=["4b", "9b"], default="9b", help="Model size (default: 9b)")
    parser.add_argument("--strategy", choices=["qlora", "full"], default="qlora", help="Training strategy (default: qlora)")
    default_region = resolve_region()
    parser.add_argument(
        "--region",
        default=default_region,
        help=(
            f"AWS region (default: {default_region}; resolved from boto3 "
            f"session, falling back to {DEFAULT_REGION_FALLBACK} when "
            f"nothing is configured)"
        ),
    )
    parser.add_argument("--role", default=ROLE_ARN, help="SageMaker execution role ARN")
    parser.add_argument("--dataset-s3", default=DATASET_S3_URI, help="S3 URI to dataset JSONL")
    parser.add_argument("--dataset-local", default=LOCAL_DATASET_PATH, help="Local dataset path to upload")
    parser.add_argument("--wait", action="store_true", help="Wait for training job to complete")
    parser.add_argument(
        "--inference-handler",
        default=None,
        help=(
            "Local path to a SageMaker inference.py. When set, the file is "
            "staged into the source bundle and copied into the merged-model "
            "directory at the end of training, so it ships inside model.tar.gz "
            "as code/inference.py — no post-job repacking required."
        ),
    )
    parser.add_argument(
        "--inference-requirements",
        default=None,
        help=(
            "Optional local requirements.txt to ship next to the inference "
            "handler. The HuggingFace Inference DLC pip-installs it on "
            "container start."
        ),
    )
    return parser.parse_args()


def stage_source_with_handler(sagemaker_code_dir, inference_handler, inference_requirements):
    """
    Copy `sagemaker_code/` into a tempdir and drop the inference handler
    (and optional requirements.txt) under `sm_inference/` — the path that
    `sm_accelerate_train.sh` looks up at the end of training.

    Returns (staged_dir, cleanup_callable). Caller must invoke cleanup
    after the SageMaker train() call returns.
    """
    if not os.path.isfile(inference_handler):
        raise FileNotFoundError(f"inference handler not found: {inference_handler}")
    if inference_requirements and not os.path.isfile(inference_requirements):
        raise FileNotFoundError(f"inference requirements not found: {inference_requirements}")

    staged_dir = tempfile.mkdtemp(prefix="qwen35-sft-src-")
    try:
        staged_code = os.path.join(staged_dir, "sagemaker_code")
        shutil.copytree(sagemaker_code_dir, staged_code)

        sm_inference_dir = os.path.join(staged_code, "sm_inference")
        os.makedirs(sm_inference_dir, exist_ok=True)
        shutil.copy2(inference_handler, os.path.join(sm_inference_dir, "inference.py"))
        if inference_requirements:
            shutil.copy2(
                inference_requirements,
                os.path.join(sm_inference_dir, "requirements.txt"),
            )
    except BaseException:
        # If staging fails partway through, the caller never receives the
        # cleanup callable, so reclaim the tempdir here before propagating.
        shutil.rmtree(staged_dir, ignore_errors=True)
        raise

    def _cleanup():
        shutil.rmtree(staged_dir, ignore_errors=True)

    return staged_code, _cleanup


def main():
    args = parse_args()
    recipe_path, instance_type = RECIPES[(args.model, args.strategy)]

    # Session
    boto_session = boto3.Session(region_name=args.region)
    sess = Session(boto_session=boto_session)
    bucket = sess.default_bucket()
    role = args.role or get_execution_role(sagemaker_session=sess)

    print(f"Region:   {args.region}")
    print(f"Role:     {role}")
    print(f"Bucket:   {bucket}")
    print(f"Recipe:   {recipe_path}")
    print(f"Instance: {instance_type}")

    # Dataset — use provided S3 URI or upload local file
    if args.dataset_s3:
        dataset_s3_uri = args.dataset_s3
        print(f"\nUsing existing dataset: {dataset_s3_uri}")
    else:
        local_path = args.dataset_local
        if not os.path.exists(local_path):
            raise FileNotFoundError(
                f"Dataset not found at {local_path}. Either:\n"
                f"  1. Place your JSONL file at {local_path}\n"
                f"  2. Pass --dataset-s3 s3://bucket/path/to/dataset.jsonl\n"
                f"  3. Pass --dataset-local /path/to/your/dataset.jsonl"
            )
        s3_key = f"{S3_PREFIX}/dataset/{os.path.basename(local_path)}"
        s3_client = boto_session.client("s3")
        print(f"\nUploading {local_path} to s3://{bucket}/{s3_key}...")
        s3_client.upload_file(local_path, bucket, s3_key)
        dataset_s3_uri = f"s3://{bucket}/{s3_key}"
        print(f"Uploaded to: {dataset_s3_uri}")

    # Training image
    # Note: DLC account ID varies by region. 763104351884 is for us-east-1/us-west-2.
    # For ap-southeast-1, use: 763104351884 (same for most commercial regions).
    # Full list: https://docs.aws.amazon.com/sagemaker/latest/dg/ecr-paths.html
    pytorch_image_uri = (
        f"763104351884.dkr.ecr.{args.region}.amazonaws.com"
        f"/pytorch-training:{DLC_TAG}"
    )
    print(f"Image:    {pytorch_image_uri}")

    # Source code
    sagemaker_code_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "sagemaker_code",
    )

    cleanup_staged_source = None
    if args.inference_handler:
        sagemaker_code_dir, cleanup_staged_source = stage_source_with_handler(
            sagemaker_code_dir,
            args.inference_handler,
            args.inference_requirements,
        )
        print(f"Sideload: {args.inference_handler} -> sm_inference/inference.py")
        if args.inference_requirements:
            print(
                f"Sideload: {args.inference_requirements} -> sm_inference/requirements.txt"
            )

    source_code = SourceCode(
        source_dir=sagemaker_code_dir,
        command=f"bash sm_accelerate_train.sh --config {recipe_path}",
    )

    compute = Compute(
        instance_type=instance_type,
        instance_count=1,
        volume_size_in_gb=200,
    )

    model_tag = f"qwen35-{args.model}"
    base_job_name = f"{model_tag}-{args.strategy}-sft"
    output_path = f"s3://{bucket}/{S3_PREFIX}/{base_job_name}"

    training_env = {"NCCL_DEBUG": "INFO"}
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        training_env["HF_TOKEN"] = hf_token
        print("HF_TOKEN: found")
    else:
        print("HF_TOKEN: not set (Qwen3.5-Base models are not gated, so this is fine)")

    model_trainer = ModelTrainer(
        training_image=pytorch_image_uri,
        source_code=source_code,
        base_job_name=base_job_name,
        compute=compute,
        stopping_condition=StoppingCondition(max_runtime_in_seconds=86400),
        output_data_config=OutputDataConfig(s3_output_path=output_path),
        # Leaving CheckpointConfig.s3_uri unset is deliberate: SDK v3 then
        # auto-derives a per-run path of the form
        # s3://<default_bucket>/<default_prefix>/<base_job_name>/<TrainingJobName>/checkpoints
        # (see sagemaker.train.model_trainer:614-618 and
        #  sagemaker.core.training.configs.CheckpointConfig docstring).
        # Pinning s3_uri here previously made every run share one
        # checkpoint folder, so SageMaker auto-restored optimizer state
        # from a prior (LoRA-shape-mismatched) run and crashed with a
        # tensor-shape RuntimeError on resume.
        checkpoint_config=CheckpointConfig(local_path="/opt/ml/checkpoints"),
        role=role,
        environment=training_env,
    )

    print(f"\nLaunching training job: {base_job_name}")
    try:
        model_trainer.train(
            input_data_config=[
                InputData(
                    channel_name="training",
                    data_source=dataset_s3_uri,
                )
            ],
            wait=args.wait,
        )
    finally:
        # ModelTrainer.train() uploads source_dir synchronously to S3
        # before issuing CreateTrainingJob, regardless of `wait`. By the
        # time we reach this finally clause the tempdir is no longer
        # needed (success or failure), so reclaiming it here is safe.
        if cleanup_staged_source is not None:
            cleanup_staged_source()

    # Echo the SDK-resolved identifiers and S3 paths so users can grep
    # for them later (e.g. to find model.tar.gz, checkpoints, or join
    # against MLflow / CloudWatch).
    #
    # _latest_training_job is a Pydantic PrivateAttr on ModelTrainer
    # (sagemaker/train/model_trainer.py:252) populated unconditionally
    # right after CreateTrainingJob in SAGEMAKER_TRAINING_JOB mode
    # (sagemaker/train/model_trainer.py:798), regardless of wait=. The
    # SDK also mutates checkpoint_config.s3_uri in place during train()
    # at lines 614-618 of the same file, so the value is visible on the
    # instance handle we already hold.
    latest = getattr(model_trainer, "_latest_training_job", None)
    job_name = getattr(latest, "training_job_name", None)
    if job_name:
        out_prefix = output_path.rstrip("/")
        ckpt_uri = getattr(
            getattr(model_trainer, "checkpoint_config", None), "s3_uri", None
        )
        print(f"\nTrainingJobName: {job_name}")
        print(f"Output prefix:   {out_prefix}/{job_name}/output/")
        print(f"Model artifact:  {out_prefix}/{job_name}/output/model.tar.gz")
        print(f"Checkpoints:     {ckpt_uri}")

    if args.wait:
        print("\nTraining job completed.")
    else:
        print("\nTraining job submitted! Monitor in SageMaker console.")


if __name__ == "__main__":
    main()
