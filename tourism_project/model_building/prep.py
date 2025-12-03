# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# # Define constants for the dataset and output paths
# api = HfApi(token=os.getenv("HF_TOKEN_MLOPS_PAT"))
# os.environ['HF_TOKEN'] = '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'
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/siaese/tourism-package-prediction/tourism.csv"

# df = pd.read_csv(DATASET_PATH)
# print("Dataset loaded successfully.")

df = pd.read_csv(DATASET_PATH, index_col=0)
print("Dataset loaded successfully.")

replacements = {
    'MaritalStatus': {'Unmarried': 'Single'},
    'Gender': {'Fe Male': 'Female'},
    'Occupation': {'Free Lancer': 'Freelancer'}
}
for col, mapping in replacements.items():
    df[col] = df[col].replace(mapping)

# Drop the unique Customer Id
df = df.drop('CustomerID', axis=1)

# Drop the unique Customer Id
# df.drop(columns=['CustomerID'], inplace=True)

# label_encoder = LabelEncoder()

# # Encoding the categorical 'TypeofContact' column
# df['TypeofContact'] = label_encoder.fit_transform(df['TypeofContact'])

# # Encoding the categorical 'Occupation' column
# # label_encoder = LabelEncoder()
# df['Occupation'] = label_encoder.fit_transform(df['Occupation'])

# # Encoding the categorical 'Gender' column
# # label_encoder = LabelEncoder()
# df['Gender'] = label_encoder.fit_transform(df['Gender'])

# # Encoding the categorical 'ProductPitched' column
# # label_encoder = LabelEncoder()
# df['ProductPitched'] = label_encoder.fit_transform(df['ProductPitched'])

# # Encoding the categorical 'MaritalStatus' column
# # label_encoder = LabelEncoder()
# df['MaritalStatus'] = label_encoder.fit_transform(df['MaritalStatus'])

# # Encoding the categorical 'Designation' column
# # label_encoder = LabelEncoder()
# df['Designation'] = label_encoder.fit_transform(df['Designation'])

encoders = {}
categorical_cols = ['TypeofContact', 'Occupation', 'Gender', 'ProductPitched', 'MaritalStatus', 'Designation']

for col in categorical_cols:
    encoders[col] = LabelEncoder()
    df[col] = encoders[col].fit_transform(df[col])

target_col = 'ProdTaken'

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Xtrain.to_csv("Xtrain.csv",index=False)
# Xtest.to_csv("Xtest.csv",index=False)
# ytrain.to_csv("ytrain.csv",index=False)
# ytest.to_csv("ytest.csv",index=False)
Xtrain.to_csv("tourism_project/data/Xtrain.csv",index=False)
Xtest.to_csv("tourism_project/data/Xtest.csv",index=False)
ytrain.to_csv("tourism_project/data/ytrain.csv",index=False)
ytest.to_csv("tourism_project/data/ytest.csv",index=False)

# files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]
files = [
    "tourism_project/data/Xtrain.csv",
    "tourism_project/data/Xtest.csv",
    "tourism_project/data/ytrain.csv",
    "tourism_project/data/ytest.csv",
]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="siaese/tourism-package-prediction",
        repo_type="dataset",
    )

print("Preprocessing complete!")
