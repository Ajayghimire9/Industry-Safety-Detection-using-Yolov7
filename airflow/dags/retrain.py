from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator

with DAG(
    "safetyvision_retraining",
    start_date=datetime(2026, 1, 1),
    schedule="@weekly",
    catchup=False,
    tags=["mlops", "computer-vision"],
) as dag:
    train = BashOperator(
        task_id="train_and_track",
        bash_command="python -m safetyvision.train --command 'python yolov7/train.py --data data/safety.yaml --weights weights/yolov7.pt --epochs 50'",
    )
    evaluate = BashOperator(
        task_id="evaluate_and_register",
        bash_command="python scripts/evaluate_model.py --weights artifacts/best.pt --manifest artifacts/model.json",
    )
    train >> evaluate
