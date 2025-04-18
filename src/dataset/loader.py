"""
loader.py

This module provides functionality for loading and processing the whiskey dataset.
It includes a class for managing the dataset and fetching associated images.

Classes:
    SpiritType: An enumeration of spirit types.
    DatasetLoader: A class for loading the dataset and fetching images.

Usage:
    loader = DatasetLoader(data_folder="data")
    loader.fetch_images()
    data = loader.load_dataset()
"""

import os
import pandas as pd
import requests
from enum import Enum

# Define the SpiritType enum
class SpiritType(Enum):
    """
    An enumeration of spirit types available in the dataset.
    """
    GINS = 'Gins'
    BOURBON = 'Bourbon'
    RYE = 'Rye'
    SCOTCH = 'Scotch'
    WHISKEY = 'Whiskey'
    LIQUEURS = 'Liqueurs'
    TEQUILA = 'Tequila'
    VODKA = 'Vodka'
    CANADIAN_WHISKY = 'Canadian Whisky'
    IRISH_WHISKEY = 'Irish Whiskey'
    SINGLE_MALT_SCOTCH = 'Single Malt Scotch Whisky'
    JAPANESE_WHISKY = 'Japanese Whisky'
    BLENDED_WHISKY = 'Blended Whisky'
    
# Define the dataset loader class
class DatasetLoader:
    """
    A class for loading and processing the whiskey dataset.

    Attributes:
        data_folder (str): The path to the data folder containing the dataset and images.

    Methods:
        fetch_images(): Fetches images from URLs in the dataset and saves them locally.
        load_dataset(): Loads the dataset as a pandas DataFrame, excluding the URL column.
    """

    def __init__(self, data_folder: str = "data"):
        """
        Initializes the DatasetLoader with the data folder path.

        Args:
            data_folder (str): The path to the data folder containing the dataset and images. Defaults to 'data'.
        """
        self.data_folder = data_folder
        self.dataset_path = os.path.join(data_folder, "dataset.tsv")
        

    def fetch_images(self):
        """
        Fetches images from the URLs in the dataset and saves them in the specified directory.

        This method iterates through the dataset, retrieves each image from its URL,
        and saves it locally with the ID as the filename.

        Raises:
            Exception: If an image cannot be fetched, an error message is printed.
        """
        image_dir = os.path.join(self.data_folder, "source_images")
        os.makedirs(image_dir, exist_ok=True)
        data = pd.read_csv(self.dataset_path, sep='\t')
        for _, row in data.iterrows():
            image_url = row['image_url']
            image_id = row['id']
            image_path = os.path.join(image_dir, f"{image_id}.png")
            try:
                response = requests.get(image_url, stream=True)
                if response.status_code == 200:
                    with open(image_path, 'wb') as f:
                        for chunk in response.iter_content(1024):
                            f.write(chunk)
            except Exception as e:
                print(f"Failed to fetch image {image_id}: {e}")

    def load_dataset(self):
        """
        Loads the dataset as a pandas DataFrame, excluding the URL column.

        Returns:
            pandas.DataFrame: The dataset without the URL column.
        """
        data = pd.read_csv(self.dataset_path, sep='\t')
        return data.drop(columns=['image_url'])