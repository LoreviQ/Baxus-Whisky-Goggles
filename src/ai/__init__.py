import pandas as pd
from enum import Enum
from algorithmic_matching import find_best_match

class Methods(Enum):
    """
    An enumeration of matching methods.
    """
    ALGORITHMIC = 'algorithmic'


def find_match(dataframe: pd.DataFrame, ocr_text:str, method:Methods=Methods.ALGORITHMIC) -> tuple[pd.Series, float]:
    """
    Find the best match for the given OCR text in the dataframe using the specified method.

    Args:
        dataframe (pd.DataFrame): The dataframe containing the data to search.
        ocr_text (str): The OCR text to match.
        method (Methods): The matching method to use. Defaults to Methods.ALGORITHMIC.

    Returns:
        tuple[pd.Series, float]: A tuple containing the best matching row and its similarity score.
    """
    if method == Methods.ALGORITHMIC:
        best_match, score = find_best_match(dataframe, ocr_text)
        if best_match is not None:
            return best_match, score
        else:
            print("No match found.")
            return None, 0.0
    return None, 0.0