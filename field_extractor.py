from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoConfig
from huggingface_hub import snapshot_download
import os

HF_MODEL_ID = "psabhay2003/t5_invoice_model"
LOCAL_MODEL_DIR = "./t5_invoice_model"

if not os.path.exists(LOCAL_MODEL_DIR) or not os.path.exists(os.path.join(LOCAL_MODEL_DIR, "pytorch_model.bin")):
    snapshot_download(HF_MODEL_ID, local_dir=LOCAL_MODEL_DIR, local_dir_use_symlinks=False)

# Use the correct model base
model_name = "t5-small"

tokenizer = T5Tokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)

model = T5ForConditionalGeneration.from_pretrained(
    pretrained_model_name_or_path=LOCAL_MODEL_DIR,
    config=config,
    local_files_only=True
)


def extract_invoice_fields(raw_text: str) -> dict:
    """Extract structured invoice fields from raw OCR text using T5 model"""
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

    result = {}
    for item in decoded_output.split(","):
        if ":" in item:
            key, val = item.split(":", 1)
            result[key.strip()] = val.strip()

    return result
