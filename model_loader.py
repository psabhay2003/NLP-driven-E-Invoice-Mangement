from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

# Your actual fine-tuned model (trained from t5-base, not t5-small)
HF_MODEL_NAME = "psabhay2003/t5_invoice_model"

def load_model_and_tokenizer():
    print(f"📦 Loading model from Hugging Face: {HF_MODEL_NAME}")
    
    model = T5ForConditionalGeneration.from_pretrained(HF_MODEL_NAME)
    tokenizer = T5Tokenizer.from_pretrained(HF_MODEL_NAME)

    # Force CPU usage (e.g., for Railway or Hugging Face Spaces)
    device = torch.device("cpu")
    model.to(device)

    print(f"✅ Loaded model with d_model = {model.config.d_model}")
    return model, tokenizer
