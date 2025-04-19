import numpy as np
import pytesseract
from preprocess import greyscale, threshold, denoise, deskew
import cv2
from typing import Tuple
import os
import csv

def process_images(source_dir: str, processed_dir: str, output_csv: str):
    """
    Iterates over images in source_dir, processes them, saves the processed
    image to processed_dir, and saves extracted text to a CSV file.

    Args:
        source_dir (str): Path to the directory containing source images.
        processed_dir (str): Path to the directory to save processed images.
        output_csv (str): Path to the output CSV file for extracted text.
    """
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    results = []
    image_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')

    print(f"Processing images from: {source_dir}")
    for filename in os.listdir(source_dir):
        if filename.lower().endswith(image_extensions):
            source_path = os.path.join(source_dir, filename)
            processed_path = os.path.join(processed_dir, filename)

            print(f"  Processing {filename}...")
            image = cv2.imread(source_path)
            if image is None:
                print(f"    Error reading {filename}. Skipping.")
                continue

            try:
                processed_image, extracted_text = process_and_extract(image)
                cv2.imwrite(processed_path, processed_image)
                csv_safe_text = extracted_text.strip().replace('\n', '\\n')
                results.append([filename, csv_safe_text])
                print(f"    Saved processed image to {processed_path}")
                print(f"    Extracted text: '{extracted_text[:50].strip()}...'")
            except Exception as e:
                print(f"    Error processing {filename}: {e}")

    print(f"\nSaving extracted text to: {output_csv}")
    try:
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['filename', 'text'])  # Write header
            writer.writerows(results)
        print("CSV file saved successfully.")
    except IOError as e:
        print(f"Error writing CSV file: {e}")


def process_and_extract(image: np.ndarray) -> Tuple[np.ndarray, str]:
    """
    Process the image and extract text using Tesseract OCR.
    Args:
        image (np.ndarray): The input image.
    Returns:
        str: The extracted text.
    """
    image = greyscale(image)
    image = denoise(image)
    image = deskew(image)
    image = threshold(image)
    text = extract_text_from_image(image)
    return image, text

def extract_text_from_image(image: np.ndarray) -> str:
    """
    Extract text from an image using Tesseract OCR.

    Args:
        image (np.ndarray): The input image in which to extract text.

    Returns:
        str: The extracted text.
    """
    pytesseract_config = r'--psm 4'
    return pytesseract.image_to_string(image, config=pytesseract_config)
    
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) # Assumes script is in src/ocr
    source_images_dir = os.path.join(base_dir, 'data', 'source_images')
    processed_images_dir = os.path.join(base_dir, 'data', 'processed_images')
    extracted_text_csv = os.path.join(base_dir, 'data', 'extracted_text.csv')

    process_images(source_images_dir, processed_images_dir, extracted_text_csv)