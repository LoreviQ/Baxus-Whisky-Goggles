from PIL import Image
import pytesseract

image_path = 'data/source_images/429.png'

try:
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    print("Extracted Text:")
    print(text)
except FileNotFoundError:
    print(f"Error: Image not found at {image_path}")
except pytesseract.TesseractNotFoundError:
    print("Error: Tesseract is not installed or not in your PATH.")
    print("Please ensure Tesseract is installed and the 'tesseract' command is working.")
    print("You might need to set the 'tesseract_cmd' variable in the script.")
except Exception as e:
    print(f"An error occurred: {e}")