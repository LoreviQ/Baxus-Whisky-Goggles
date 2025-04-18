import cv2
import numpy as np
import os

test_image = 'data/augmented_images/429_3.png'


def greyscale(image: np.ndarray) -> np.ndarray:
    """
    Convert an image to grayscale.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)




if __name__ == "__main__":
    # Example usage
    image = cv2.imread(test_image)
    if image is not None:
        gray_image = greyscale(image)
        cv2.imwrite('gray_image.png', gray_image)
    else:
        print(f"Error: Image not found at {test_image}")