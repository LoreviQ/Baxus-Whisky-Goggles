import numpy as np
import pytesseract
from preprocess import greyscale, threshold, denoise, deskew
import cv2

test_image = 'data/source_images/53003.png'

def extract_text_from_image(image: np.ndarray) -> str:
    """
    Extract text from an image using Tesseract OCR.

    Args:
        image (np.ndarray): The input image in which to extract text.

    Returns:
        str: The extracted text.
    """
    pytesseract_config = r'--psm 4'

    image = greyscale(image)
    image = denoise(image)
    image = deskew(image)
    image = threshold(image)
    text = pytesseract.image_to_string(image, config=pytesseract_config)
    image = cv2.flip(image, 1, image)
    flip_text = pytesseract.image_to_string(image, config=pytesseract_config)
    if text and not flip_text:
        return text
    elif flip_text and not text:
        return flip_text
    elif text and flip_text:
        return text + "\n" + flip_text
    else:
        return "No text found in the image."
    

if __name__ == "__main__":
    # Example usage
    image = cv2.imread(test_image)
    if image is not None:
        text = extract_text_from_image(image)
        print("Extracted Text:")
        print(text)
    else:
        print(f"Error: Image not found at {test_image}")