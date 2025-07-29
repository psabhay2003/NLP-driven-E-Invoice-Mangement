import os
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from safetensors.torch import load_file
import requests

class InvoiceFieldExtractor:
    def __init__(self, model_dir="./model", model_file="model.safetensors", hf_model_repo="psabhay2003/t5_invoice_model"):
        self.model_path = os.path.join(model_dir, model_file)
        self.tokenizer_path = model_dir
        self.hf_model_repo = hf_model_repo

        if not os.path.exists(self.model_path):
            print("🔍 Local model file not found. Downloading from Hugging Face...")
            self.download_model()

        print("🧠 Loading tokenizer...")
        self.tokenizer = T5Tokenizer.from_pretrained(self.tokenizer_path)

        print("⚡ Loading quantized model from safetensors...")
        config = T5Config.from_pretrained(self.hf_model_repo)
        state_dict = load_file(self.model_path)
        self.model = T5ForConditionalGeneration(config)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def download_model(self):
        os.makedirs(self.tokenizer_path, exist_ok=True)
        model_url = f"https://huggingface.co/{self.hf_model_repo}/resolve/main/model.safetensors"
        response = requests.get(model_url)
        if response.status_code == 200:
            with open(self.model_path, 'wb') as f:
                f.write(response.content)
            print("✅ model.safetensors downloaded.")
        else:
            raise RuntimeError(f"Failed to download model from: {model_url}")

    def extract_fields(self, ocr_text):
        input_text = f"extract fields: {ocr_text.strip().replace('\\n', ' ')}"
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = self.model.generate(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self.parse_result(result)

    def parse_result(self, result_text):
        fields = {}
        for item in result_text.split(';'):
            if ':' in item:
                key, value = item.split(':', 1)
                fields[key.strip()] = value.strip()
        return fields
