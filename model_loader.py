from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

# Replace this with your actual Hugging Face model repo
HF_MODEL_NAME = "psabhay2003/t5_invoice_model"  # your fine-tuned model

def load_model_and_tokenizer():
    print(f"📦 Loading model from Hugging Face: {HF_MODEL_NAME}")
    model = T5ForConditionalGeneration.from_pretrained(HF_MODEL_NAME)
    tokenizer = T5Tokenizer.from_pretrained(HF_MODEL_NAME)

    # 🚫 No GPU on Railway – force CPU
    device = torch.device("cpu")
    model.to(device)
    
    print("🚀 Model and tokenizer loaded successfully.")
    return model, tokenizer
