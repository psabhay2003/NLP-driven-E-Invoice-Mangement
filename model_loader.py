from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

HF_MODEL_NAME = "psabhay2003/t5_invoice_model"

def load_model_and_tokenizer():
    print(f"📦 Loading model from Hugging Face: {HF_MODEL_NAME}")
    model = AutoModelForSeq2SeqLM.from_pretrained(HF_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)

    device = torch.device("cpu")
    model.to(device)

    print("🚀 Model and tokenizer loaded successfully on CPU.")
    return model, tokenizer
