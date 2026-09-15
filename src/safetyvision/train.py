from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import mlflow


def run_training(command: str, run_name: str = "safetyvision-training") -> None:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("training_command", command)
        completed = subprocess.run(command, shell=True, check=False, text=True)
        mlflow.log_metric("process_exit_code", completed.returncode)
        if completed.returncode != 0:
            raise RuntimeError(f"Training failed with exit code {completed.returncode}")


def register_artifact(weights: str, metrics: dict, output: str) -> None:
    from .registry import create_manifest
    create_manifest(weights, metrics, output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLOv7 training under MLflow")
    parser.add_argument("--command", required=True)
    args = parser.parse_args()
    run_training(args.command)


if __name__ == "__main__":
    main()
