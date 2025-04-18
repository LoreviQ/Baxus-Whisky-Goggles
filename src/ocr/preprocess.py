import cv2
import numpy as np
import os

test_image = 'data/augmented_images/429_3.png'


def greyscale(image: np.ndarray) -> np.ndarray:
    """
    Convert an image to grayscale.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# upscale TDOD

def threshold(image: np.ndarray, block_size: int = 11, C: int = 2) -> np.ndarray:
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
    # Blurring before adaptive thresholding can sometimes reduce noise artifacts
    # blurred_image = cv2.medianBlur(gray_image, 3) 
    binary_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, C)
    return binary_image

if __name__ == "__main__":
    # Example usage
    image = cv2.imread(test_image)
    if image is not None:
        gray_image = greyscale(image)
        cv2.imwrite('gray_image.png', gray_image)
        threshold_image = threshold(gray_image)
        cv2.imwrite('threshold_image.png', threshold_image)
    else:
        print(f"Error: Image not found at {test_image}")