def extract_fields(text, model, tokenizer):
    input_ids = tokenizer(text, return_tensors='pt', truncation=True).input_ids
    output = model.generate(input_ids)
    extracted_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return eval(extracted_text) if '{' in extracted_text else {'error': 'Failed to parse'}