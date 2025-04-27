"""
Used for tuning parameters of the OCR model.
"""
import cv2
from preprocess import greyscale, threshold, denoise, deskew
from upscale import upscale
from tesseract import extract_text_from_image

def tune_adaptive_threshold(image_path: str) -> None:
    image = cv2.imread(image_path)
    image = greyscale(image)
    image = upscale(image)

    block_sizes = [11, 15, 21, 31, 41, 51, 71, 91, 111, 151]
    C_values = [0, 2, 5, 7, -1, -3, -5, -7, -11, -15, -19, -23]
    for block_size in block_sizes:
        for C in C_values:
            print(f"Testing: block_size={block_size}, C={C}")
            thresholded_image = threshold(image, block_size=block_size, C=C)
            cv2.imwrite(f"test_images/threshold_block_{block_size}_C_{C}.png", thresholded_image)
            text = extract_text_from_image(thresholded_image)
            print(f"Extracted text: {text.strip()}")
            print("-" * 50)

if __name__ == "__main__":
    image_path = 'data/source_images/16367.png'
    tune_adaptive_threshold(image_path)
