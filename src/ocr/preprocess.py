import cv2
import math
import numpy as np
from deskew import determine_skew
from typing import Tuple, Union

test_image = 'data/source_images/53003.png'

def greyscale(image: np.ndarray) -> np.ndarray:
    """
    Convert an image to grayscale.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def threshold(image: np.ndarray, block_size: int = 111, C: int = -3) -> np.ndarray:
    """
    Apply adaptive Gaussian thresholding to an image.

    Args:
        image (np.ndarray): The input image.
        block_size (int): Size of the pixel neighborhood that is used to calculate a threshold value for the pixel.
                          It must be an odd number. Default is 11.
        C (int): A constant subtracted from the mean or weighted mean. Default is 2.
    Returns:
        np.ndarray: The binary image after applying adaptive thresholding.
    """
    binary_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, C)
    return binary_image

def denoise(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Apply Gaussian blur to reduce noise in the image.

    Args:
        image (np.ndarray): The input image.
        kernel_size (int): Size of the Gaussian kernel. Default is 3. Must be odd and greater than 1.
    Returns:
        np.ndarray: The denoised image.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    if kernel_size <= 1:
        kernel_size = 3
    return cv2.medianBlur(image, kernel_size)

def denoise_nlm(image: np.ndarray, h: float = 10.0, 
                 templateWindowSize: int = 7, searchWindowSize: int = 21) -> np.ndarray:
      """Apply Non-Local Means Denoising."""
      # Ensure image is uint8, required by NLM
      if image.dtype != np.uint8:
           # Basic scaling assuming input might be float 0-1 or other ranges
          if image.max() <= 1.0 and image.min() >= 0.0:
              image = (image * 255).astype(np.uint8)
          else:
              # Perform robust normalization if needed, or just cast
              image = np.clip(image, 0, 255).astype(np.uint8)

      return cv2.fastNlMeansDenoising(image, None, h, templateWindowSize, searchWindowSize)

def deskew(image: np.ndarray) -> np.ndarray:
    """
    Deskew the image to correct for any tilt.

    Args:
        image (np.ndarray): The input image.
    Returns:
        np.ndarray: The deskewed image.
    """
    angle = determine_skew(image)
    return rotate(image, angle)

def rotate(
        image: np.ndarray, angle: float, background: Union[int, Tuple[int, int, int]] = (255, 255, 255)
) -> np.ndarray:
    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
    height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2
    return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background)

# potential future processing steps
# Advanced Adaptive Thresholding (varying params)
# Lighting Normalization
# rembg
# glare removal
# curvature correction
# italics
# edge detection
