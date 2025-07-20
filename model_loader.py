import os
import requests
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

MODEL_DIR = "model"
MODEL_FILE = "model.safetensors"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)

# Your Google Drive file ID
FILE_ID = "15jfduaYqSLjFshjKDcw5W6F0cBcfvB8L"

def download_from_google_drive(file_id, dest_path):
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

    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

    print(f"Downloaded model to {dest_path}")

def load_model():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    if not os.path.exists(MODEL_PATH):
        print("Model not found locally. Downloading from Google Drive...")
        download_from_google_drive(FILE_ID, MODEL_PATH)

    model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)
    tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
    return model, tokenizer
