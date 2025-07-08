from PIL import Image
import pytesseract

def extract_text_from_image(image_path: str) -> str:
    """Extract raw text from an invoice image using Tesseract OCR"""
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    return text
