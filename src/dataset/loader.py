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
from enum import Enum
from .images import fetch_images

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
        dataset (pandas.DataFrame): The loaded dataset.

    Methods:
        fetch_images(): Fetches images from the dataset and saves them locally.
        load_dataset(): Loads the dataset as a pandas DataFrame.
    """

    def __init__(self, data_folder: str = "data"):
        """
        Initializes the DatasetLoader with the data folder path and loads the dataset.

        Args:
            data_folder (str): The path to the data folder containing the dataset and images. Defaults to 'data'.
        """
        self.data_folder = data_folder
        self.dataset = self.load_dataset()
        self._fetch_images()

    def load_dataset(self, dataset_path: str = "dataset.tsv") -> pd.DataFrame:
        """
        Loads the dataset as a pandas DataFrame and sets it to the dataset attribute.

        Args:
            dataset_path (str): The path to the dataset file. Defaults to 'dataset.tsv'.
        """
        full_path = os.path.join(self.data_folder, dataset_path)
        return pd.read_csv(full_path, sep='\t')
    
    def _fetch_images(self):
        """
        Fetches images from the dataset and saves them locally.
        """
        image_dir = os.path.join(self.data_folder, "source_images")
        fetch_images(image_dir, self.dataset)
        self.dataset['path'] = self.dataset.apply(
            lambda row: os.path.join(image_dir, f"{row['id']}.png"), axis=1
        )
        self.dataset.drop(columns=['image_url'], inplace=True)