from huggingface_hub import HfApi, create_repo
import os

# Define the Hugging Face Space repository ID
# Replace 'your-username' with your actual Hugging Face username
HF_SPACE_REPO_ID = "pjhansi2404/Wellness_Tourism_Predictor_Space"

# Initialize Hugging Face API
api = HfApi(token=os.getenv("HF_TOKEN"))

# Path to your deployment files
DEPLOYMENT_FOLDER = "tourism_project/deployment"

def deploy_to_hf_space():
    print(f"Attempting to create/get Hugging Face Space: {HF_SPACE_REPO_ID}")
    try:
        # Create a new Space or use an existing one
        create_repo(
            repo_id=HF_SPACE_REPO_ID,
            repo_type="space",
            space_sdk="streamlit", # Specify Streamlit SDK
            private=False, # Set to True if you want a private Space
            exist_ok=True # Allow creation if it already exists
        )
        print(f"Hugging Face Space '{HF_SPACE_REPO_ID}' ensured (created or found).")

        # Upload the contents of the deployment folder to the Space
        print(f"Uploading deployment files from '{DEPLOYMENT_FOLDER}' to '{HF_SPACE_REPO_ID}'...")
        api.upload_folder(
            folder_path=DEPLOYMENT_FOLDER,
            repo_id=HF_SPACE_REPO_ID,
            repo_type="space",
            commit_message="Initial deployment of Streamlit app"
        )
        print(f"Deployment files successfully uploaded to Hugging Face Space: {HF_SPACE_REPO_ID}")
        print(f"You can view your Space at: https://huggingface.co/spaces/{HF_SPACE_REPO_ID}")

    except Exception as e:
        print(f"An error occurred during deployment to Hugging Face Space: {e}")

if __name__ == "__main__":
    # Ensure HF_TOKEN is set in your environment variables
    if os.getenv("HF_TOKEN") is None:
        print("Error: HF_TOKEN environment variable is not set. Please set it before running.")
    else:
        deploy_to_hf_space()
