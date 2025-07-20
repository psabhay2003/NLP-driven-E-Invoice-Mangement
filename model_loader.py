import os
import requests
from transformers import T5Tokenizer, T5ForConditionalGeneration, SafetensorsCheckpoint
import torch

# Configuration
MODEL_DIR = "model"
MODEL_FILE = "model.safetensors"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)

# Replace with your actual Google Drive file ID
FILE_ID = "15jfduaYqSLjFshjKDcw5W6F0cBcfvB8L"

def download_from_google_drive(file_id, dest_path):
    """Download large files from Google Drive with token handling."""
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                return value
        return None

    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

    print(f"[INFO] Downloaded model file to: {dest_path}")

def load_model_and_tokenizer():
    """Load model and tokenizer, downloading the model if missing."""
    if not os.path.exists(MODEL_PATH):
        print("[INFO] Model file not found locally. Downloading...")
        download_from_google_drive(FILE_ID, MODEL_PATH)

    print("[INFO] Loading tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)

    print("[INFO] Loading model from safetensors...")
    model = T5ForConditionalGeneration.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        use_safetensors=True
    )

    return model, tokenizer
