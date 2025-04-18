import os
import pandas as pd
import logging
from rembg import remove
import requests

logger = logging.getLogger("BaxusLogger")

def fetch_images(images_directory: str, dataframe: pd.DataFrame):
    """
    Fetches images from the URLs in the provided dataframe and saves them in the specified directory.

    Args:
        images_directory (str): The directory to save the images.
        dataframe (pd.DataFrame): The dataframe containing image URLs and IDs.

    Raises:
        Exception: If an image cannot be fetched, an error message is printed.
    """
    logger.info(f"Fetching images")
    os.makedirs(images_directory, exist_ok=True)
    for _, row in dataframe.iterrows():
        image_url = row['image_url']
        image_id = row['id']
        image_path = os.path.join(images_directory, f"{image_id}.png")
        if os.path.exists(image_path):
            logger.info(f"Image {image_id} already exists. Skipping download.")
            continue
        try:
            response = requests.get(image_url, stream=True)
            if response.status_code == 200:
                with open(image_path, 'wb') as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)
        except Exception as e:
            logger.warning(f"Failed to fetch image {image_id}: {e}")

def remove_image_backgrounds(source_directory: str, destination_directory: str):
    """
    Removes the background from all PNG images in the source directory using rembg and saves the processed images
    with the same name in the destination directory. Skips images that already exist in the destination directory.

    Args:
        source_directory (str): Directory containing the source images.
        destination_directory (str): Directory to save the processed images.
    """

    logger.info(f"Removing bakgrounds with rembg")
    os.makedirs(destination_directory, exist_ok=True)
    for filename in os.listdir(source_directory):
        if filename.endswith('.png'):
            source_path = os.path.join(source_directory, filename)
            destination_path = os.path.join(destination_directory, filename)

            if os.path.exists(destination_path):
                logger.info(f"Image {filename} already exists in the destination directory. Skipping.")
                continue

            try:
                with open(source_path, "rb") as input_file:
                    with open(destination_path, "wb") as output_file:
                        input_data = input_file.read()
                        output_data = remove(input_data)
                        output_file.write(output_data)
            except Exception as e:
                logger.warning(f"Failed to process image {filename}: {e}")