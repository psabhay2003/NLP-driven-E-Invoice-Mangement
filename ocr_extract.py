import easyocr

reader = None

def extract_text_from_image(image_path: str) -> str:
    """Extract raw text from an invoice image using EasyOCR"""
    global reader
    if reader is None:
        reader = easyocr.Reader(['en'], gpu=False)
    results = reader.readtext(image_path, detail=0)
    return "\n".join(results)
