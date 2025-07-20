from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

# Your Hugging Face model repo
HF_MODEL_NAME = "psabhay2003/t5_invoice_model"  # your fine-tuned T5-small model

def load_model_and_tokenizer():
    print(f"📦 Loading model from Hugging Face: {HF_MODEL_NAME}")
    
    try:
        # Load tokenizer
        tokenizer = T5Tokenizer.from_pretrained(HF_MODEL_NAME)
        
        # Load model with size validation
        model = T5ForConditionalGeneration.from_pretrained(
            HF_MODEL_NAME,
            ignore_mismatched_sizes=False  # Set to True only if you expect size differences
        )

        # Force model to run on CPU (important for Railway/Spaces)
        device = torch.device("cpu")
        model.to(device)
        
        # Optional sanity check
        print(f"✅ Model architecture: d_model={model.config.d_model} (should be 512 for t5-small)")
        assert model.config.d_model == 512, "🚨 This model is not based on t5-small. Please check the source."

        print("🚀 Model and tokenizer loaded successfully.")
        return model, tokenizer

    except Exception as e:
        print(f"❌ Error loading model or tokenizer: {e}")
        raise e
