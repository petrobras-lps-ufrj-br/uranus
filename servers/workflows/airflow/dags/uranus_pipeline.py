from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

def collect_data():
    print("Collecting data from Cognite...")

def process_data():
    print("Processing data with Uranus AI...")

def deploy_model():
    print("Deploying model to Triton Server...")

default_args = {
    'owner': 'uranus',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'uranus_pipeline',
    default_args=default_args,
    description='A simple Uranus AI pipeline',
    schedule_interval=timedelta(days=1),
    catchup=False,
) as dag:

    t1 = PythonOperator(
        task_id='collect_data',
        python_callable=collect_data,
    )

    t2 = PythonOperator(
        task_id='process_data',
        python_callable=process_data,
    )

    t3 = PythonOperator(
        task_id='deploy_model',
        python_callable=deploy_model,
    )

    t1 >> t2 >> t3
