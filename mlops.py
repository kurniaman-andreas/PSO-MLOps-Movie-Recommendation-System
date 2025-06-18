# import pandas as pd
# from surprise import Dataset, Reader, SVD, accuracy
# import pickle
# import os
# import mlflow
# import logging
# from steps_model.preprocess_and_split import preprocess_and_split
# from itertools import product

# # Setup logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

# def main():
#     logging.info("Starting SVD training pipeline with Grid Search...")

#     os.makedirs("mlruns", exist_ok=True)
#     mlflow.set_tracking_uri("file:mlruns")
#     mlflow.set_experiment("SVD Movie Recommendation")

#     # Step 1: Preprocessing
#     preprocess_and_split()
#     logging.info("✅ Preprocessing done.")

#     # Step 2: Load train & test data
#     train_df = pd.read_csv('data/train.csv')
#     test_df = pd.read_csv('data/test.csv')

#     reader = Reader(rating_scale=(1, 5))
#     train_data = Dataset.load_from_df(train_df[['user_id', 'product_id', 'rating']], reader)
#     trainset = train_data.build_full_trainset()
#     testset = list(zip(test_df['user_id'], test_df['product_id'], test_df['rating']))

#     # Grid search parameters
#     param_grid = {
#         'n_factors': [50, 100],
#         'lr_all': [0.002, 0.005],
#         'reg_all': [0.02, 0.1]
#     }

#     best_rmse = float('inf')
#     best_params = {}
#     best_model = None

#     # Iterate through all combinations
#     for n_factors, lr_all, reg_all in product(param_grid['n_factors'], param_grid['lr_all'], param_grid['reg_all']):
#         with mlflow.start_run(nested=True):
#             model = SVD(n_factors=n_factors, lr_all=lr_all, reg_all=reg_all)
#             model.fit(trainset)
#             predictions = model.test(testset)
#             rmse = accuracy.rmse(predictions, verbose=False)

#             mlflow.log_param("n_factors", n_factors)
#             mlflow.log_param("lr_all", lr_all)
#             mlflow.log_param("reg_all", reg_all)
#             mlflow.log_metric("RMSE", rmse)

#             if rmse < best_rmse:
#                 best_rmse = rmse
#                 best_params = {"n_factors": n_factors, "lr_all": lr_all, "reg_all": reg_all}
#                 best_model = model
#                 best_predictions = predictions

#     # Final run to save best model
#     with mlflow.start_run(run_name="best_model"):
#         mlflow.set_tag("developer", "kurniaman-andreas")
#         mlflow.set_tag("model_type", "SVD with GridSearch")
#         mlflow.log_params(best_params)
#         mlflow.log_metric("RMSE", best_rmse)

#         # Save model
#         model_dir = os.path.join(".", "artifacts", "model")
#         os.makedirs(model_dir, exist_ok=True)
#         model_filename = os.path.join(model_dir, "svd_model.pkl")
#         with open(model_filename, "wb") as f:
#             pickle.dump(best_model, f)

#         mlflow.log_artifact(model_filename, artifact_path="model")
#         logging.info(f"✅ Best model saved with params: {best_params} and RMSE: {best_rmse:.4f}")

# if __name__ == "__main__":
#     main()
import pandas as pd
from surprise import Dataset, Reader, SVD, accuracy
import pickle
import os
import wandb
import logging
from steps_model.preprocess_and_split import preprocess_and_split
from itertools import product

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

def main():
    logging.info("🚀 Starting SVD training pipeline with Grid Search using wandb...")

    # Step 1: Preprocessing
    preprocess_and_split()
    logging.info("✅ Preprocessing done.")

    # Step 2: Load train & test data
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')

    reader = Reader(rating_scale=(1, 5))
    train_data = Dataset.load_from_df(train_df[['user_id', 'product_id', 'rating']], reader)
    trainset = train_data.build_full_trainset()
    testset = list(zip(test_df['user_id'], test_df['product_id'], test_df['rating']))

    # Grid search parameters
    param_grid = {
        'n_factors': [50, 100],
        'lr_all': [0.002, 0.005],
        'reg_all': [0.02, 0.1]
    }

    best_rmse = float('inf')
    best_params = {}
    best_model = None
    best_predictions = None

    # Iterate through all combinations
    for n_factors, lr_all, reg_all in product(param_grid['n_factors'], param_grid['lr_all'], param_grid['reg_all']):
        # Start new wandb run
        run = wandb.init(
            project="svd-recommender",
            config={
                "n_factors": n_factors,
                "lr_all": lr_all,
                "reg_all": reg_all
            },
            reinit=True  # allow multiple runs in same script
        )

        model = SVD(n_factors=n_factors, lr_all=lr_all, reg_all=reg_all)
        model.fit(trainset)
        predictions = model.test(testset)
        rmse = accuracy.rmse(predictions, verbose=False)

        wandb.log({"RMSE": rmse})

        logging.info(f"Run with n_factors={n_factors}, lr_all={lr_all}, reg_all={reg_all} → RMSE={rmse:.4f}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_params = {"n_factors": n_factors, "lr_all": lr_all, "reg_all": reg_all}
            best_model = model
            best_predictions = predictions

        wandb.finish()

    # Final best model run
    final_run = wandb.init(project="svd-recommender", name="best_model", config=best_params)
    final_run.tags = ["best_model", "gridsearch", "svd"]
    wandb.log({"best_rmse": best_rmse})
    logging.info(f"✅ Best model saved with params: {best_params} and RMSE: {best_rmse:.4f}")

    # Save the model
    model_dir = os.path.join(".", "model")  # Ganti dari "artifacts/model" ke "model"
    os.makedirs(model_dir, exist_ok=True)
    model_filename = os.path.join(model_dir, "svd_model.pkl")
    with open(model_filename, "wb") as f:
        pickle.dump(best_model, f)

    # Upload model to wandb
    wandb.save(model_filename)

    wandb.finish()

if __name__ == "__main__":
    main()
