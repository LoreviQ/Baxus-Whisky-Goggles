
import numpy as np
import logging
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact 

logger = logging.getLogger("BaxusLogger")

def upscale_image(image: np.ndarray, scale=4, tile_size=0, use_gpu=True) -> np.ndarray:
    """
    Upscales an image using Real-ESRGAN.

    Args:
        image (np.ndarray): input image.
        scale (int): The upscale factor (usually determined by the model, e.g., 4 for x4 models).
        tile_size (int): Tile size for processing large images to save memory.
                         0 means no tiling. Recommended values: 256, 512.
        use_gpu (bool): Whether to use GPU if available.
    """
    logger.info(f"Model: RealESRGAN_x4plus, Scale: {scale}, Tile: {tile_size}, GPU: {use_gpu}")

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
            model_path=None,
            model=model,
            dni_weight=None,
            tile=tile_size,
            tile_pad=10,
            pre_pad=0,
            half=fp16,
            gpu_id=gpu_id
        )
    except Exception as e:
        print(f"Error initializing RealESRGANer: {e}")
        print("Ensure you have run 'pip install realesrgan basicsr' and have internet connection for model download.")
        return

    # --- Perform Upscaling ---
    try:
        output_img, _ = upsampler.enhance(image, outscale=scale) 
    except RuntimeError as error:
        print('Error during enhancement:', error)
        print('If out of memory, try increasing tile_size (e.g., 256 or 512) or disable GPU (use_gpu=False).')
        return
    except Exception as e:
        print(f'An unexpected error occurred during enhancement: {e}')
        return

    return output_img