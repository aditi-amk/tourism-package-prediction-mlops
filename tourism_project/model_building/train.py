
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, classification_report, recall_score
# for model serialization
import joblib
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow

# Set the tracking URL for MLflow
# mlflow.set_tracking_uri("http://localhost:3000")
mlflow.set_tracking_uri("https://handleable-unvenomed-karole.ngrok-free.dev")


# Set the name for the experiment
mlflow.set_experiment("mlops-training-experiment")

# Hugging Face API token from environment variable
# HF_TOKEN = os.getenv("HF_TOKEN_MLOPS_PAT")
# api = HfApi(token=HF_TOKEN)
# api = HfApi()

# os.environ['HF_TOKEN'] = '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'
api = HfApi(token=os.getenv("HF_TOKEN"))

# Load datasets from Hugging Face dataset repo
# Xtrain_path = "https://huggingface.co/datasets/siaese/tourism-package-prediction/blob/main/Xtrain.csv"
# Xtest_path = "https://huggingface.co/datasets/siaese/tourism-package-prediction/blob/main/Xtest.csv"
# ytrain_path = "https://huggingface.co/datasets/siaese/tourism-package-prediction/blob/main/ytrain.csv"
# ytest_path = "https://huggingface.co/datasets/siaese/tourism-package-prediction/blob/main/ytest.csv"

Xtrain_path = "https://huggingface.co/datasets/siaese/tourism-package-prediction/resolve/main/Xtrain.csv"
Xtest_path = "https://huggingface.co/datasets/siaese/tourism-package-prediction/resolve/main/Xtest.csv"
ytrain_path = "https://huggingface.co/datasets/siaese/tourism-package-prediction/resolve/main/ytrain.csv"
ytest_path = "https://huggingface.co/datasets/siaese/tourism-package-prediction/resolve/main/ytest.csv"

# Xtrain_path = "hf://datasets/siaese/tourism-package-prediction/Xtrain.csv"
# Xtest_path = "hf://datasets/siaese/tourism-package-prediction/Xtest.csv"
# ytrain_path = "hf://datasets/siaese/tourism-package-prediction/ytrain.csv"
# ytest_path = "hf://datasets/siaese/tourism-package-prediction/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)

# Features for preprocessing
numeric_features = [
    'Age', 'CityTier', 'DurationOfPitch', 'NumberOfPersonVisiting', 'NumberOfFollowups',
    'PreferredPropertyStar', 'NumberOfTrips', 'Passport', 'PitchSatisfactionScore',
    'OwnCar', 'NumberOfChildrenVisiting', 'MonthlyIncome'
]
categorical_features = [
    'TypeofContact', 'Occupation', 'Gender', 'ProductPitched', 'MaritalStatus', 'Designation'
]

# Calculate scale_pos_weight for imbalance handling
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# Preprocessing pipeline: scale numeric, one-hot encode categorical
# preprocessor = make_column_transformer(
#     (StandardScaler(), numeric_features),
#     (OneHotEncoder(handle_unknown='ignore'), categorical_features)
# )

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# Define XGBoost classifier pipeline
# xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

param_grid = {
    'xgbclassifier__n_estimators': [50, 75],
    'xgbclassifier__max_depth': [2, 3],
    'xgbclassifier__colsample_bytree': [0.4, 0.5],
    'xgbclassifier__colsample_bylevel': [0.4, 0.5],
    'xgbclassifier__learning_rate': [0.01, 0.05],
    'xgbclassifier__reg_lambda': [0.4, 0.5],
}

model_pipeline = make_pipeline(preprocessor, xgb_model)

with mlflow.start_run():
    # Hyperparameter tuning
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    # Log all parameter combinations and their mean test scores
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]
        std_score = results['std_test_score'][i]

        # Log each combination as a separate MLflow run
        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_score", mean_score)
            mlflow.log_metric("std_test_score", std_score)

    # Log best parameters separately in main run
    mlflow.log_params(grid_search.best_params_)

    # Store and evaluate the best model
    best_model = grid_search.best_estimator_

    classification_threshold = 0.45

    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1-score": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1-score": test_report['1']['f1-score']
    })

    # Save the model locally
    model_path = "best_tourism_project_model_v1.joblib"
    # model_path = "tourism_project/deployment/best_tourism_project_model_v1.joblib"
    joblib.dump(best_model, model_path)

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # Upload to Hugging Face
    repo_id = "siaese/tourism-package-prediction"
    repo_type = "model"

    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Space '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Space '{repo_id}' not found. Creating new space...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Space '{repo_id}' created.")

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=model_path,
        repo_id=repo_id,
        repo_type=repo_type,
    )

# grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring='recall', n_jobs=-1)
# grid_search.fit(Xtrain, ytrain.values.ravel())

# best_model = grid_search.best_estimator_
# print("Best Params:\n", grid_search.best_params_)

# y_pred_train = best_model.predict(Xtrain)
# y_pred_test = best_model.predict(Xtest)

# print("\nTraining Tourism Classification Report:")
# print(classification_report(ytrain, y_pred_train))

# print("\nTest Tourism Classification Report:")
# print(classification_report(ytest, y_pred_test))

# # Save model file locally
# model_file = "best_wellness_tourism_model_v1.joblib"
# joblib.dump(best_model, model_file)

# # Upload model to Hugging Face Model Hub
# repo_id = "siaese/wellness_tourism_model"
# repo_type = "model"

# # Hugging Face API token from environment variable
# HF_TOKEN = os.getenv("HF_TOKEN_MLOPS_PAT")
# api = HfApi(token=HF_TOKEN)
# # api = HfApi(token=os.getenv("HF_TOKEN"))

# try:
#     api.repo_info(repo_id=repo_id, repo_type=repo_type)
#     print(f"HF Model repo '{repo_id}' exists. Using it.")
# except RepositoryNotFoundError:
#     print(f"HF Model repo '{repo_id}' not found. Creating repo...")
#     create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
#     print(f"Created model repo '{repo_id}'.")

# api.upload_file(
#     path_or_fileobj=model_file,
#     path_in_repo=model_file,
#     repo_id=repo_id,
#     repo_type=repo_type
# )

# print("Model training and registration completed successfully.")
