"""
Launch SageMaker Training Job for Qwen3.5 SFT
Uploads a local JSONL dataset to S3, then starts a training job.
SageMaker Python SDK v3 (ModelTrainer)

Usage:
    python launch_sft_job.py                                    # defaults: 9B QLoRA
    python launch_sft_job.py --model 4b --strategy qlora        # 4B QLoRA
    python launch_sft_job.py --model 9b --strategy full         # 9B full fine-tuning

Prerequisites:
    pip install sagemaker boto3 datasets
"""
import argparse
import os
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
REGION = "ap-southeast-1"
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


def parse_args():
    parser = argparse.ArgumentParser(description="Launch Qwen3.5 SFT on SageMaker")
    parser.add_argument("--model", choices=["4b", "9b"], default="9b", help="Model size (default: 9b)")
    parser.add_argument("--strategy", choices=["qlora", "full"], default="qlora", help="Training strategy (default: qlora)")
    parser.add_argument("--region", default=REGION, help=f"AWS region (default: {REGION})")
    parser.add_argument("--role", default=ROLE_ARN, help="SageMaker execution role ARN")
    parser.add_argument("--dataset-s3", default=DATASET_S3_URI, help="S3 URI to dataset JSONL")
    parser.add_argument("--dataset-local", default=LOCAL_DATASET_PATH, help="Local dataset path to upload")
    parser.add_argument("--wait", action="store_true", help="Wait for training job to complete")
    return parser.parse_args()


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
        checkpoint_config=CheckpointConfig(
            s3_uri=os.path.join(output_path, "checkpoints"),
            local_path="/opt/ml/checkpoints",
        ),
        role=role,
        environment=training_env,
    )

    print(f"\nLaunching training job: {base_job_name}")
    model_trainer.train(
        input_data_config=[
            InputData(
                channel_name="training",
                data_source=dataset_s3_uri,
            )
        ],
        wait=args.wait,
    )

    if args.wait:
        print("\nTraining job completed.")
    else:
        print("\nTraining job submitted! Monitor in SageMaker console.")


if __name__ == "__main__":
    main()
