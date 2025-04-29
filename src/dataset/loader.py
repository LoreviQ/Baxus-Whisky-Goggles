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
from .images import (
    fetch_images,
    remove_image_backgrounds,
    add_backgrounds_to_images,
    augment_images,
)
from .fuzzymatch import find_best_matches


# Define the SpiritType enum
class SpiritType(Enum):
    """
    An enumeration of spirit types available in the dataset.
    """

    GINS = "Gins"
    BOURBON = "Bourbon"
    RYE = "Rye"
    SCOTCH = "Scotch"
    WHISKEY = "Whiskey"
    LIQUEURS = "Liqueurs"
    TEQUILA = "Tequila"
    VODKA = "Vodka"
    CANADIAN_WHISKY = "Canadian Whisky"
    IRISH_WHISKEY = "Irish Whiskey"
    SINGLE_MALT_SCOTCH = "Single Malt Scotch Whisky"
    JAPANESE_WHISKY = "Japanese Whisky"
    BLENDED_WHISKY = "Blended Whisky"


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
        self.OCR_data = self.dataset[
            ["id", "name", "size", "proof", "abv", "spirit_type"]
        ].copy()

    def load_dataset(self, dataset_path: str = "dataset.tsv") -> pd.DataFrame:
        """
        Loads the dataset as a pandas DataFrame and sets it to the dataset attribute.

        Args:
            dataset_path (str): The path to the dataset file. Defaults to 'dataset.tsv'.
        """
        full_path = os.path.join(self.data_folder, dataset_path)
        return pd.read_csv(full_path, sep="\t")

    def _fetch_images(self):
        """
        Fetches images from the dataset and saves them locally.
        """
        image_dir = os.path.join(self.data_folder, "source_images")
        fetch_images(image_dir, self.dataset)
        self.dataset.drop(columns=["image_url"], inplace=True)

    def _remove_background(self):
        """
        Removes the background from images using rembg and saves them in a new directory.
        """
        source_dir = os.path.join(self.data_folder, "source_images")
        dest_dir = os.path.join(self.data_folder, "no_bg_images")
        remove_image_backgrounds(source_dir, dest_dir)

    def _add_backgrounds(self):
        """
        Adds backgrounds to images with transparent backgrounds.
        """
        source_dir = os.path.join(self.data_folder, "no_bg_images")
        background_dir = os.path.join(self.data_folder, "bgs")
        add_backgrounds_to_images(self.data_folder, source_dir, background_dir)

    def process_images(self, n=1):
        """
        Processes the images by applying albumentations pipeline and saving them.
        Args:
            n (int): The number of times to repeat the image processing. Defaults to 1.
        """
        self._remove_background()
        self._add_backgrounds()
        source_dir = os.path.join(self.data_folder, "bg_images")
        for i in range(n):
            augment_images(self.data_folder, source_dir)

    def process_ocr_data(self):
        for col in ["id", "size_ml", "proof", "abv"]:
            if col in self.OCR_data.columns:
                self.OCR_data[col] = pd.to_numeric(self.OCR_data[col], errors="coerce")
        for col in ["name", "spirit_type"]:
            if col in self.OCR_data.columns:
                self.OCR_data[col] = self.OCR_data[col].astype(str).str.lower()

    def get_best_matches(self, ocr_text: str, limit: int = 5) -> pd.DataFrame:
        """
        Finds the best matching rows in the DataFrame for a given OCR text string
        using fuzzy matching on text columns and returns the matched rows with scores.

        Args:
            ocr_text (str): The text extracted from OCR.
            limit (int): The maximum number of matches to return. Defaults to 5.

        Returns:
            pd.DataFrame: A DataFrame containing the best matching rows from the
                          original dataset, with an added 'ocr_score' column.
                          Returns an empty DataFrame if no matches are found.
        """
        matches = find_best_matches(ocr_text, self.OCR_data, limit)

        if not matches:
            # Return an empty DataFrame with original columns + score
            return pd.DataFrame(columns=list(self.dataset.columns) + ["ocr_score"])

        # Extract IDs and scores
        match_ids = [match[0] for match in matches]
        scores_map = {match[0]: match[1] for match in matches}

        # Filter the original dataset based on matched IDs
        filtered_df = self.dataset[self.dataset["id"].isin(match_ids)].copy()

        # Add the score column
        filtered_df["ocr_score"] = filtered_df["id"].map(scores_map)

        # Sort the DataFrame by score in descending order
        filtered_df.sort_values(by="ocr_score", ascending=False, inplace=True)

        return filtered_df
