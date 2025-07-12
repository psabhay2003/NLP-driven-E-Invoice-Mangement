from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load model and tokenizer from Hugging Face Hub
MODEL_REPO = "psabhay2003/t5_invoice_model"  # Replace with your actual repo path

tokenizer = T5Tokenizer.from_pretrained(MODEL_REPO)
model = T5ForConditionalGeneration.from_pretrained(MODEL_REPO)

def extract_invoice_fields(raw_text: str) -> dict:
    """Extract structured invoice fields from raw OCR text"""
    input_text = f"Extract invoice fields: {raw_text}"
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512
    )

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
