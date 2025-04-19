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

logger = logging.getLogger("BaxusLogger")

def upscale_image(image: np.ndarray, model_path: str = 'models/RealESRGAN_x4plus.pth', scale=4, tile_size=0, use_gpu=True) -> np.ndarray:
    """
    Upscales an image using Real-ESRGAN.

    Args:
        image (np.ndarray): input image.
        scale (int): The upscale factor (usually determined by the model, e.g., 4 for x4 models).
        tile_size (int): Tile size for processing large images to save memory.
                         0 means no tiling. Recommended values: 256, 512.
        use_gpu (bool): Whether to use GPU if available.
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
        return

    return output_img