import os
import pandas as pd
import logging
from rembg import remove
import requests
from PIL import Image

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

def add_backgrounds_to_images(data_directory: str, source_directory: str, background_directory: str):
    """
    Adds backgrounds to images with transparent backgrounds.

    Args:
        data_directory (str): The base directory to save the composite images.
        source_directory (str): Directory containing images with transparent backgrounds.
        background_directory (str): Directory containing background images.
    """
    logger.info("Adding backgrounds to images")

    # Create the base directory for background images
    bg_images_base_dir = os.path.join(data_directory, "bg_images")
    os.makedirs(bg_images_base_dir, exist_ok=True)

    # Iterate over each background image
    for bg_filename in os.listdir(background_directory):
        bg_path = os.path.join(background_directory, bg_filename)
        if not os.path.isfile(bg_path):
            continue

        # Create a directory for the current background
        bg_output_dir = os.path.join(bg_images_base_dir, os.path.splitext(bg_filename)[0])
        os.makedirs(bg_output_dir, exist_ok=True)

        # Open the background image
        try:
            background = Image.open(bg_path)
        except Exception as e:
            logger.warning(f"Failed to open background {bg_filename}: {e}")
            continue

        # Iterate over each source image
        for source_filename in os.listdir(source_directory):
            source_path = os.path.join(source_directory, source_filename)
            output_path = os.path.join(bg_output_dir, source_filename)

            # Skip processing if the composite image already exists
            if os.path.exists(output_path):
                logger.info(f"Composite image {output_path} already exists. Skipping.")
                continue

            if not source_filename.endswith('.png') or not os.path.isfile(source_path):
                continue

            # Open the source image
            try:
                source_image = Image.open(source_path).convert("RGBA")
            except Exception as e:
                logger.warning(f"Failed to open source image {source_filename}: {e}")
                continue

            # Resize the background to match the source image height
            try:
                bg_width, bg_height = background.size
                src_width, src_height = source_image.size
                new_bg_width = int(bg_width * (src_height / bg_height))
                resized_background = background.resize((new_bg_width, src_height))
            except Exception as e:
                logger.warning(f"Failed to resize background {bg_filename} for {source_filename}: {e}")
                continue

            # Composite the source image on top of the background
            try:
                composite = Image.new("RGBA", (src_width, src_height))
                composite.paste(resized_background, (0, 0))
                composite.paste(source_image, (0, 0), source_image)

                # Save the composite image
                composite.save(output_path, "PNG")
            except Exception as e:
                logger.warning(f"Failed to create composite for {source_filename} with {bg_filename}: {e}")