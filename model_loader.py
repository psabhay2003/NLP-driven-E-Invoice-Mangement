from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

HF_MODEL_NAME = "psabhay2003/t5_invoice_model"  # Trained from t5-base

def load_model_and_tokenizer():
    print(f"📦 Loading model from Hugging Face: {HF_MODEL_NAME}")
    
    model = T5ForConditionalGeneration.from_pretrained(HF_MODEL_NAME)
    tokenizer = T5Tokenizer.from_pretrained(HF_MODEL_NAME)

    device = torch.device("cpu")
    model.to(device)
    
    print(f"✅ Loaded model with d_model = {model.config.d_model} (should be 768 for t5-base)")
    return model, tokenizer
