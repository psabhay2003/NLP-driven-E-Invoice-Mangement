import os
import requests
from transformers import T5Tokenizer, T5ForConditionalGeneration

def download_model():
    model_dir = "t5_invoice_model"
    model_file = os.path.join(model_dir, "model.safetensors")

    if not os.path.exists(model_file):
        os.makedirs(model_dir, exist_ok=True)
        print("Downloading model...")
        url = "https://your-direct-link.com/model.safetensors"  # ← Replace this
        response = requests.get(url)
        with open(model_file, "wb") as f:
            f.write(response.content)

        # Do the same for other required files (config.json, tokenizer_config.json, etc.)
        # You can repeat for each file
        # Or zip them all and unzip here
download_model()

# Load model and tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5_invoice_model")
model = T5ForConditionalGeneration.from_pretrained("t5_invoice_model")


# Load model and tokenizer once
tokenizer = T5Tokenizer.from_pretrained("./t5_invoice_model")  # Adjust path
model = T5ForConditionalGeneration.from_pretrained("./t5_invoice_model")

def extract_invoice_fields(raw_text: str) -> dict:
    """Extract structured invoice fields from raw OCR text"""
    input_text = f"Extract invoice fields: {raw_text}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)

    output = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=128
    )

    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

    # Post-process into dictionary
    result = {}
    for item in decoded_output.split(","):
        if ":" in item:
            key, val = item.split(":", 1)
            result[key.strip()] = val.strip()
    return result
