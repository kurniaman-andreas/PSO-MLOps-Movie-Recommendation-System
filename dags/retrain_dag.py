from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import subprocess
import logging

default_args = {
    'owner': 'hajid',
    'depends_on_past': False,
    'start_date': datetime(2025, 6, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'retrain_svd_movie',
    default_args=default_args,
    description='Retrain SVD Movie Recommendation model daily',
    schedule_interval='@daily',
    catchup=False,
)

def run_retraining():
    logging.info("Running MLOps retraining script...")
    subprocess.run(["python", "/opt/airflow/mlops.py"], check=True)


retrain_task = PythonOperator(
    task_id='retrain_svd',
    python_callable=run_retraining,
    dag=dag
)
