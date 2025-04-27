"""
Upscale an image using Real-ESRGAN.

Contains a patch script for basicsr and realesrgan to ensure compatibility with the latest torchvision version.
"""

# ----- BEGIN TORCHVISION PATCH -----
import importlib
import sys
import types
import torchvision
import torchvision.transforms.functional as F_vision # Import the functional module where functions now reside

# Try to find the specification for the old module path
functional_tensor_spec = importlib.util.find_spec('torchvision.transforms.functional_tensor')

# If the spec is None, it means the module path doesn't exist (likely newer torchvision)
if functional_tensor_spec is None:
    print("Patching torchvision: 'functional_tensor' module not found. Attempting to create a mock.")
    try:
        # Check if the required function exists in the new location (torchvision.transforms.functional)
        if hasattr(F_vision, 'rgb_to_grayscale'):
            # Create a new, empty module object named 'functional_tensor'
            mock_functional_tensor = types.ModuleType('torchvision.transforms.functional_tensor')

            # Get the actual rgb_to_grayscale function from its new location
            rgb_to_grayscale_func = getattr(F_vision, 'rgb_to_grayscale')

            # Add the function to our mock module
            setattr(mock_functional_tensor, 'rgb_to_grayscale', rgb_to_grayscale_func)

            # Inject the mock module into sys.modules AND into the torchvision.transforms namespace
            # This makes it findable by Python's import system
            sys.modules['torchvision.transforms.functional_tensor'] = mock_functional_tensor
            setattr(torchvision.transforms, 'functional_tensor', mock_functional_tensor)

            print(f"Patch successful: 'rgb_to_grayscale' function injected into mock 'torchvision.transforms.functional_tensor'")
        else:
            print("Patch warning: 'rgb_to_grayscale' not found in 'torchvision.transforms.functional'. Cannot apply patch.")

    except ImportError:
        # Handle cases where even 'torchvision.transforms.functional' might not be importable
        print("Patch error: Could not import 'torchvision.transforms.functional'. Cannot apply patch.")
    except Exception as e:
        print(f"Patch error: An unexpected error occurred during patching: {e}")

else:
    # The module exists, likely an older torchvision or a different setup. No patch needed.
    print("No patch needed: 'torchvision.transforms.functional_tensor' module found.")

# ----- END TORCHVISION PATCH -----

import numpy as np
import logging
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import os
import cv2

logger = logging.getLogger("BaxusLogger")

def upscale(image: np.ndarray, model_path: str = 'models/RealESRGAN_x4plus.pth', scale=4, tile_size=0, use_gpu=True) -> np.ndarray | None: # <-- Added None to return type hint
    """
    Upscales an image using Real-ESRGAN.

    Args:
        image (np.ndarray): input image.
        model_path (str): Path to the Real-ESRGAN model file.
        scale (int): The upscale factor (usually determined by the model, e.g., 4 for x4 models).
        tile_size (int): Tile size for processing large images to save memory.
                         0 means no tiling. Recommended values: 256, 512.
        use_gpu (bool): Whether to use GPU if available.

    Returns:
        np.ndarray | None: The upscaled image as a NumPy array, or None if an error occurred.
    """

    # --- Validate Model Path ---
    if not os.path.isfile(model_path):
        logger.error(f"Model file not found at: {model_path}")
        logger.error("Please download the model (e.g., RealESRGAN_x4plus.pth) and place it at the specified path.")
        return None
    
    model_name = os.path.basename(model_path)
    logger.info(f"Model: {model_name}, Scale: {scale}, Tile: {tile_size}, GPU: {use_gpu}")

    # --- Determine Device ---
    if use_gpu and torch.cuda.is_available():
        gpu_id = 0
        fp16 = True
        logger.info("Using GPU.")
    else:
        gpu_id = None
        fp16 = False
        logger.info("Using CPU.")


    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    # --- Set up the RealESRGANer ---
    try:
        upsampler = RealESRGANer(
            scale=scale,
            model_path=model_path,
            model=model,
            dni_weight=None,
            tile=tile_size,
            tile_pad=10,
            pre_pad=0,
            half=fp16,
            gpu_id=gpu_id
        )
        logger.info("RealESRGANer initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing RealESRGANer: {e}")
        logger.error("Ensure you have run 'pip install realesrgan basicsr' and have internet connection for model download.")
        return

    # --- Perform Upscaling ---
    try:
        output_img, _ = upsampler.enhance(image, outscale=scale)
        logger.info("Image upscaled successfully.")
    except RuntimeError as error:
        logger.error('Error during enhancement:', error)
        logger.error('If out of memory, try increasing tile_size (e.g., 256 or 512) or disable GPU (use_gpu=False).')
        return
    except Exception as e:
        logger.error(f'An unexpected error occurred during enhancement: {e}')
        return None # <-- Return None on error

    return output_img

# --- Main Execution Block ---
if __name__ == "__main__":
    source_dir = 'data/source_images'
    upscale_dir = 'data/upscaled_images'
    model_file = 'models/RealESRGAN_x4plus.pth'

    # Ensure the output directory exists
    os.makedirs(upscale_dir, exist_ok=True)
    logger.info(f"Ensured output directory exists: {upscale_dir}")

    # Check if the model file exists before starting the loop
    if not os.path.isfile(model_file):
        logger.error(f"FATAL: Model file not found at {model_file}. Please download it.")
        exit(1) # Exit if the model isn't found

    # List all files in the source directory
    try:
        files = os.listdir(source_dir)
    except FileNotFoundError:
        logger.error(f"FATAL: Source directory not found: {source_dir}")
        exit(1)
    except Exception as e:
        logger.error(f"FATAL: Error listing files in {source_dir}: {e}")
        exit(1)

    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')
    logger.info(f"Starting upscaling process from '{source_dir}' to '{upscale_dir}'...")

    for filename in files:
        # Check if the file is an image
        if not filename.lower().endswith(image_extensions):
            logger.debug(f"Skipping non-image file: {filename}")
            continue

        source_path = os.path.join(source_dir, filename)
        dest_path = os.path.join(upscale_dir, filename)

        # Skip if the destination image already exists
        if os.path.exists(dest_path):
            logger.info(f"Skipping '{filename}', already exists in '{upscale_dir}'.")
            continue

        logger.info(f"Processing '{filename}'...")

        # Load the image
        img = None
        try:
            img = cv2.imread(source_path)
            if img is None:
                logger.warning(f"Could not read image file: {source_path}. Skipping.")
                continue
        except Exception as e:
            logger.error(f"Error loading image {source_path}: {e}")
            continue

        # Upscale the image
        upscaled_img = upscale(img, model_path=model_file, use_gpu=True)

        # Save the upscaled image or the original on failure
        image_to_save = upscaled_img if upscaled_img is not None else img
        operation_type = "upscaled" if upscaled_img is not None else "original (upscaling failed)"

        # Check if the image to save is valid
        if image_to_save is None:
            logger.error(f"Cannot save image for {filename}, no valid image data available (original or upscaled).")
            continue

        # Save the image
        try:
            success = cv2.imwrite(dest_path, image_to_save)
        except Exception as e:
            logger.error(f"Error saving {operation_type} image {dest_path}: {e}")      

    logger.info(f"Upscaling process complete for directory '{source_dir}'. Check logs for details.")