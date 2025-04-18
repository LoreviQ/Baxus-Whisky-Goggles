"""
Simple script to fetch the images from the dataset.
"""

from dataset.loader import DatasetLoader

def fetch_images():
    """
    Fetches the images from the dataset.
    """

    # Initialize the dataset loader
    loader = DatasetLoader()

    # Fetch images
    print("Fetching images...")
    loader.fetch_images()
    print("Images fetched successfully.")

if __name__ == "__main__":
    fetch_images()