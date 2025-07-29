from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoConfig

# Use the correct base model
model_name = "t5-small"

# Load tokenizer and model config from the official T5-small
tokenizer = T5Tokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)

# Load your fine-tuned model from the local folder with T5-small config
model = T5ForConditionalGeneration.from_pretrained(
    pretrained_model_name_or_path="./t5_invoice_model",
    config=config,
    local_files_only=True
)

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
