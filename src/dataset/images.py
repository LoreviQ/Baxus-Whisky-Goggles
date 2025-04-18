import os
import pandas as pd
import logging
from rembg import remove
import requests
from PIL import Image
import albumentations as A
import cv2
import numpy as np

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

def augment_images(data_directory: str, source_directory: str):
    logger.info("Augmenting images")

    # Create the base directory for output images
    output_images_base_dir = os.path.join(data_directory, "augmented_images")
    os.makedirs(output_images_base_dir, exist_ok=True)

    # Walk through all subdirectories and files in the source directory
    for root, _, files in os.walk(source_directory):
        for file in files:
            if not file.endswith('.png'):
                continue

            source_path = os.path.join(root, file)
            try:
                # Read the image
                image = cv2.imread(source_path, cv2.IMREAD_UNCHANGED)
                if image is None:
                    logger.warning(f"Failed to read image {source_path}")
                    continue
                
                # Convert to RGB for Albumentations
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Apply augmentation
                augmented_image = augment_image(image)

                # Generate a unique filename for the augmented image
                base_name, ext = os.path.splitext(file)
                n = 0
                while True:
                    output_filename = f"{base_name}_{n}.png"
                    output_path = os.path.join(output_images_base_dir, output_filename)
                    if not os.path.exists(output_path):
                        break
                    n += 1

                # Save the augmented image
                cv2.imwrite(output_path, augmented_image)
                logger.info(f"Saved augmented image to {output_path}")

            except Exception as e:
                logger.warning(f"Failed to augment image {source_path}: {e}")

def augment_image(image : np.ndarray) -> np.ndarray:
    """
    Augments a single image using albumentations.
    Args:
        image (np.ndarray): The input image to augment.
    Returns:
        np.ndarray: The augmented image.
    """
    # transform pipeline
    transform = A.Compose([
        A.Rotate(limit=15, p=0.5),  # Random rotation up to 15 degrees, with 50% probability
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5), # Combined shift, scale, rotate
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.Perspective(scale=(0.05, 0.1), p=0.3), # Simulate slight perspective changes
        A.HorizontalFlip(p=0.5),
        # Add more augmentations as needed
    ])
    augmented = transform(image=image)
    return augmented['image']