
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os
# from dotenv import load_dotenv; load_dotenv()

# os.environ['HF_TOKEN'] = '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'
# Initialize API client
# api = HfApi(token=os.getenv('HF_TOKEN'))

repo_id = "siaese/tourism-package-prediction"
repo_type = "dataset"

# Initialize API client - reads HF_TOKEN from environment automatically
api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")

api.upload_folder(
    folder_path="tourism_project/data",
    repo_id=repo_id,
    repo_type=repo_type,
)

# Create repo if doesn't exist
# try:
#     print("hi")
#     api.repo_info(repo_id=repo_id, repo_type=repo_type)
#     print(f"✅ Repo '{repo_id}' exists.")
# except RepositoryNotFoundError:
#     create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
#     print(f"✅ Created repo '{repo_id}'.")

# # ✅ UPLOAD tourism.csv (your existing file)
# api.upload_file(
#     path_or_fileobj="tourism_project/data/tourism.csv",  # ← YOUR EXISTING FILE
#     path_in_repo="tourism.csv",
#     repo_id=repo_id,
#     repo_type=repo_type,
#     commit_message="Upload raw tourism dataset"
# )

# print(f"✅ Check: https://huggingface.co/datasets/{repo_id}")
