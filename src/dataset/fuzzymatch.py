import pandas as pd
from thefuzz import fuzz


def find_best_matches(ocr_text: str, data: pd.DataFrame) -> list[tuple[int, int]]:
    """
    Finds the best matching rows in the DataFrame for a given OCR text string
    using fuzzy matching on text columns.

    Args:
        ocr_text (str): The text extracted from OCR.
        data (pd.DataFrame): The DataFrame to search within.
                             Expected columns: 'id', 'name', 'spirit_type'.
                             'name' and 'spirit_type' should be preprocessed (e.g., lowercase).
        limit (int): The maximum number of matches to return. Defaults to 5.

    Returns:
        list[tuple[int, int]]: A list of tuples, where each tuple contains
                               (id, score), sorted by score in descending order.
                               Returns an empty list if the DataFrame is empty.
    """
    if data.empty:
        return []

    processed_ocr_text = ocr_text.lower().strip()
    matches = []

    # Ensure required columns exist and are of string type
    if (
        "name" not in data.columns
        or "spirit_type" not in data.columns
        or "id" not in data.columns
    ):
        raise ValueError(
            "DataFrame must contain 'id', 'name', and 'spirit_type' columns."
        )

    # Convert relevant columns to string just in case they aren't
    data_copy = data.copy()
    data_copy["name"] = data_copy["name"].astype(str)
    data_copy["spirit_type"] = data_copy["spirit_type"].astype(str)

    for row in data_copy.itertuples(index=False):
        # Combine relevant text fields for matching
        # Assumes 'name' and 'spirit_type' are already lowercased by process_ocr_data
        combined_text = f"{row.name} {row.spirit_type}"

        # Calculate fuzzy score
        # token_sort_ratio handles out-of-order words well
        score = fuzz.token_sort_ratio(processed_ocr_text, combined_text)

        matches.append((row.id, score))

    # Sort matches by score in descending order
    matches.sort(key=lambda item: item[1], reverse=True)

    return matches


# Example Usage (assuming you have a DataFrame `df`):
# df = pd.DataFrame({
#     'id': [1, 2, 3],
#     'name': ['larceny kentucky straight bourbon whiskey', 'elijah craig small batch', 'makers mark'],
#     'size': [750, 750, 750],
#     'proof': [92, 94, 90],
#     'abv': [46, 47, 45],
#     'spirit_type': ['bourbon', 'bourbon', 'bourbon']
# })
# ocr_output = '—O JUIN IIIUIOLD O—\n\nLARCENY\n\nADNTUCKT SINNUGOT BOURDON WMIsREY'
# best_matches = find_best_matches(ocr_output, df)
# print(best_matches)
