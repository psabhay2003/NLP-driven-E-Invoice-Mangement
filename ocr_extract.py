from PIL import Image
import pytesseract

def extract_text_from_image(image_path: str) -> str:
    """Extract text from image using Tesseract OCR."""
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text
