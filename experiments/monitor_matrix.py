"""
Poll SageMaker training-job status for every test row in results.json
and update the row in place. Exits when every job has reached a terminal
state (Completed | Failed | Stopped).
"""
import json
import time
import datetime
from pathlib import Path
import boto3

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = REPO_ROOT / "experiments" / "results.json"
TERMINAL = {"Completed", "Failed", "Stopped"}


def now_iso():
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def main():
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Results file not found: {RESULTS_PATH}. "
            "Run `python experiments/launch_matrix.py` first to bootstrap "
            "the matrix and submit jobs."
        )
    with open(RESULTS_PATH) as f:
        results = json.load(f)
    region = results.get("region", "us-east-1")
    sm = boto3.client("sagemaker", region_name=region)

    while True:
        all_done = True
        for row in results["tests"]:
            job = row.get("training_job_name")
            if not job:
                continue
            if row.get("final_status") in TERMINAL:
                continue
            try:
                d = sm.describe_training_job(TrainingJobName=job)
            except Exception as e:
                row["describe_error"] = repr(e)
                continue
            status = d["TrainingJobStatus"]
            row["final_status"] = status
            row["secondary_status"] = d.get("SecondaryStatus")
            row["billable_seconds"] = d.get("BillableTimeInSeconds")
            if status == "Failed":
                row["failure_reason"] = d.get("FailureReason")
            if status not in TERMINAL:
                all_done = False
        results["last_polled_utc"] = now_iso()
        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2)
        if all_done:
            break
        time.sleep(60)

    print("All jobs in terminal state.")
    for row in results["tests"]:
        print(f"  {row['id']}: {row.get('final_status')} (billable {row.get('billable_seconds')}s) {row.get('training_job_name')}")


if __name__ == "__main__":
    main()
