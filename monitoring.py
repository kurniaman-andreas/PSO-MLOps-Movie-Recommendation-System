import os
import pickle
import pandas as pd
import mlflow
from surprise import Reader, Dataset, SVD
from evidently import Report
from evidently.presets import DataDriftPreset, RegressionPreset
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

def generate_monitoring_reports():
    logging.info("🔍 Starting monitoring with Evidently...")

    # Load data
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')

    # Load model
    model_path = "model/svd_model.pkl"
    if not os.path.exists(model_path):
        logging.error("❌ Model file not found.")
        return

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Prepare Surprise testset
    testset = list(zip(test_df['user_id'], test_df['product_id'], test_df['rating']))

    # Predict
    predictions = model.test(testset)

    # Convert to DataFrame
    pred_df = pd.DataFrame(predictions)
    pred_df['true_rating'] = pred_df['r_ui']
    pred_df['pred_rating'] = pred_df['est']

    # Create reports
    os.makedirs("reports", exist_ok=True)

    # 1. Data Drift Report
    data_drift_report = Report(metrics=[DataDriftPreset()])
    my_eval = data_drift_report.run(reference_data=train_df, current_data=test_df)
    drift_path = "reports/data_drift_report.html"
    my_eval.save_html(drift_path)

    # 2. Model Performance Report
    performance_report = Report(metrics=[RegressionPreset()])
    performance_report.run(
        reference_data=pred_df[['true_rating']],
        current_data=pred_df[['pred_rating']]
    )
    performance_report_path = "reports/performance_report.html"
    performance_report.save(performance_report_path)

    # Log to MLflow
    mlflow.set_experiment("SVD Movie Recommendation")
    with mlflow.start_run(run_name="monitoring_reports"):
        mlflow.set_tag("report_type", "evidently")
        mlflow.log_artifact(drift_path, artifact_path="reports")
        mlflow.log_artifact(performance_report_path, artifact_path="reports")

    logging.info("✅ Monitoring reports generated and logged to MLflow.")

if __name__ == "__main__":
    generate_monitoring_reports()
