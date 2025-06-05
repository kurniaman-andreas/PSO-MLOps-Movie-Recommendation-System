import pandas as pd
from surprise import Dataset, Reader, SVD, accuracy
import pickle
import os
import mlflow
import logging
from steps_model.preprocess_and_split import preprocess_and_split

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

def main():
    logging.info("Starting SVD training pipeline...")

    # Start MLflow experiment
    mlflow.set_experiment("SVD Movie Recommendation")

    with mlflow.start_run() as run:
        # Step 1: Preprocessing
        preprocess_and_split()
        logging.info("✅ Preprocessing done.")

        # Step 2: Load train & test data
        train_df = pd.read_csv('data/train.csv')
        test_df = pd.read_csv('data/test.csv')

        # Step 3: Prepare data for Surprise
        reader = Reader(rating_scale=(1, 5))
        train_data = Dataset.load_from_df(train_df[['user_id', 'product_id', 'rating']], reader)
        trainset = train_data.build_full_trainset()
        testset = list(zip(test_df['user_id'], test_df['product_id'], test_df['rating']))

        # Step 4: Train model
        model = SVD(n_factors=100, lr_all=0.005, reg_all=0.02)
        model.fit(trainset)
        predictions = model.test(testset)

        # Step 5: Evaluate
        rmse = accuracy.rmse(predictions, verbose=False)
        mae = accuracy.mae(predictions, verbose=False)
        logging.info(f"✅ RMSE: {rmse:.4f}, MAE: {mae:.4f}")

        # Step 6: MLflow logging
        mlflow.set_tag("developer", "kurniaman-andreas")
        mlflow.set_tag("model_type", "SVD with surprise")

        if hasattr(model, "n_factors"):
            mlflow.log_param("n_factors", model.n_factors)
        if hasattr(model, "lr_all"):
            mlflow.log_param("lr_all", model.lr_all)
        if hasattr(model, "reg_all"):
            mlflow.log_param("reg_all", model.reg_all)

        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAE", mae)

        # Step 7: Save and log model
        model_filename = "model/svd_model.pkl"
        os.makedirs(os.path.dirname(model_filename), exist_ok=True)
        with open(model_filename, "wb") as f:
            pickle.dump(model, f)

        mlflow.log_artifact(model_filename, artifact_path="model")
        logging.info(f"✅ Model saved and logged to MLflow under: {model_filename}")

if __name__ == "__main__":
    main()
