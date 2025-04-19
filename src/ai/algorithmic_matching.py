import pandas as pd
import re
from thefuzz import fuzz
import math
import pandas as pd
from typing import Union

# --- Define Weights (CRUCIAL - needs tuning!) ---
# Assign higher values to features you trust more or are more discriminating.
# Name is often very important but also prone to OCR errors.
# ABV/Proof are often reliable if extracted correctly.
feature_weights = {
    'name': 0.5,        # High weight for name similarity
    'abv': 0.2,         # Moderate weight for ABV
    'proof': 0.1,       # Lower weight if ABV is already used
    'spirit_type': 0.1, # Moderate weight for type
    'size_ml': 0.1      # Lower weight for size
}

def extract_features_from_ocr(ocr_text: str) -> dict[str, Union[float, set, str]]:
    """
    Extracts potential features (ABV, Proof, Spirit Type, Size, Name keywords)
    from raw OCR text.
    """
    text_lower = ocr_text.lower()
    features = {
        'abv': None,
        'proof': None,
        'spirit_type_keywords': set(),
        'size_ml': None,
        'text_for_name_matching': text_lower # Start with all text
    }

    # --- Extract ABV/Proof ---
    # Look for patterns like XX%, XX % vol, XX alc, XX proof
    # More robust regex can handle variations like 47% ALC./ VOL
    abv_match = re.search(r'(\d+(\.\d+)?)\s*%\s*(alc|vol)', text_lower)
    proof_match = re.search(r'(\d+(\.\d+)?)\s*(proof)', text_lower)

    if abv_match:
        try:
            features['abv'] = float(abv_match.group(1))
            # Infer proof if ABV found and proof not explicitly found
            if not proof_match:
                 features['proof'] = features['abv'] * 2
            # Remove matched part from text used for name matching
            features['text_for_name_matching'] = features['text_for_name_matching'].replace(abv_match.group(0), '')
        except (ValueError, IndexError):
            pass # Ignore if conversion fails

    if proof_match:
        try:
            features['proof'] = float(proof_match.group(1))
            # Infer ABV if Proof found and ABV not explicitly found
            if not abv_match:
                features['abv'] = features['proof'] / 2
            # Remove matched part from text used for name matching
            features['text_for_name_matching'] = features['text_for_name_matching'].replace(proof_match.group(0), '')
        except (ValueError, IndexError):
             pass # Ignore if conversion fails

    # --- Extract Spirit Type Keywords ---
    common_types = ['bourbon', 'whiskey', 'whisky', 'scotch', 'rye', 'rum', 'vodka', 'gin', 'tequila']
    words = set(re.findall(r'\b\w+\b', text_lower))
    for type_keyword in common_types:
        if type_keyword in words:
            features['spirit_type_keywords'].add(type_keyword)
            # Optionally remove from name text - be careful not to remove too much
            # features['text_for_name_matching'] = re.sub(r'\b' + type_keyword + r'\b', '', features['text_for_name_matching'])

    # --- Extract Size (Example: 750ml, 70cl, 1l) ---
    size_match = re.search(r'(\d+(\.\d+)?)\s*(ml|cl|l)\b', text_lower)
    if size_match:
        try:
            value = float(size_match.group(1))
            unit = size_match.group(3)
            if unit == 'l':
                features['size_ml'] = value * 1000
            elif unit == 'cl':
                features['size_ml'] = value * 10
            else: # ml
                features['size_ml'] = value
            # Remove matched part from text used for name matching
            features['text_for_name_matching'] = features['text_for_name_matching'].replace(size_match.group(0), '')
        except (ValueError, IndexError):
            pass

    # --- Clean up text for name matching ---
    # Remove common noise, extra spaces, newlines etc.
    features['text_for_name_matching'] = re.sub(r'\s+', ' ', features['text_for_name_matching']).strip()
    # Consider removing generic terms if they aren't part of names
    # e.g. features['text_for_name_matching'] = features['text_for_name_matching'].replace('straight', '').replace('batch', '') # Be cautious!

    return features

def calculate_similarity(features: dict[str, Union[float, set, str]], dataset_row:pd.Series, weights: dict[str, float]):
    """
    Calculates a similarity score between extracted OCR features and a dataset row.
    Weights determine the importance of each field.
    """
    total_score = 0
    total_weight = 0

    # --- Name Similarity (using Fuzzy Matching) ---
    if 'name' in dataset_row.index and features.get('text_for_name_matching'):
        # Choose a fuzzy matching strategy. token_sort_ratio is often good for names.
        # partial_ratio might be useful if OCR only captured part of the name.
        name_score = fuzz.token_sort_ratio(features['text_for_name_matching'], dataset_row['name']) / 100.0 # Scale to 0-1
        total_score += name_score * weights.get('name', 0)
        total_weight += weights.get('name', 0)
        # print(f"Debug Name: OCR='{features['text_for_name_matching']}', DB='{dataset_row['name']}', Score={name_score}")


    # --- ABV Similarity ---
    ocr_abv = features.get('abv')
    db_abv = dataset_row.get('abv')
    if ocr_abv is not None and db_abv is not None and not math.isnan(db_abv):
        # Simple difference scoring (closer is better)
        # Max expected diff could be ~5% ABV? Normalize based on that.
        max_abv_diff = 5.0
        diff = abs(ocr_abv - db_abv)
        abv_score = max(0, 1 - (diff / max_abv_diff)) # Score = 1 if identical, 0 if diff >= max_abv_diff
        total_score += abv_score * weights.get('abv', 0)
        total_weight += weights.get('abv', 0)
        # print(f"Debug ABV: OCR={ocr_abv}, DB={db_abv}, Score={abv_score}")

    # --- Proof Similarity (similar to ABV) ---
    ocr_proof = features.get('proof')
    db_proof = dataset_row.get('proof')
    if ocr_proof is not None and db_proof is not None and not math.isnan(db_proof):
        max_proof_diff = 10.0 # e.g., 2 * max_abv_diff
        diff = abs(ocr_proof - db_proof)
        proof_score = max(0, 1 - (diff / max_proof_diff))
        total_score += proof_score * weights.get('proof', 0)
        total_weight += weights.get('proof', 0)
        # print(f"Debug Proof: OCR={ocr_proof}, DB={db_proof}, Score={proof_score}")


    # --- Spirit Type Similarity ---
    ocr_types = features.get('spirit_type_keywords', set())
    db_type = dataset_row.get('spirit_type') # Assumes lowercase
    if ocr_types and db_type and not pd.isna(db_type):
        # Score 1 if the DB type is mentioned in OCR keywords, 0 otherwise
        # Or fuzzy match db_type against the keywords? Simple check is often enough.
        type_score = 1.0 if db_type in ocr_types else 0.0
        # Could enhance: If 'whiskey' in ocr_types and db_type is 'bourbon', give partial score?
        total_score += type_score * weights.get('spirit_type', 0)
        total_weight += weights.get('spirit_type', 0)
        # print(f"Debug Type: OCR={ocr_types}, DB={db_type}, Score={type_score}")

    # --- Size Similarity (similar to ABV/Proof) ---
    ocr_size = features.get('size_ml')
    db_size = dataset_row.get('size_ml')
    if ocr_size is not None and db_size is not None and not math.isnan(db_size):
        # Sizes are often standard, so maybe a smaller tolerance?
        max_size_diff = 50.0 # e.g., difference in ml
        diff = abs(ocr_size - db_size)
        # Handle exact match or close match strongly
        if diff < 1: # Allow for minor float issues
             size_score = 1.0
        else:
             size_score = max(0, 1 - (diff / max_size_diff))

        total_score += size_score * weights.get('size_ml', 0)
        total_weight += weights.get('size_ml', 0)
        # print(f"Debug Size: OCR={ocr_size}, DB={db_size}, Score={size_score}")


    # --- Calculate Final Weighted Score ---
    if total_weight == 0:
        return 0 # Avoid division by zero if no weights matched
    final_score = total_score / total_weight
    return final_score

def find_best_match(ocr_text: str, df: pd.DataFrame, weights: dict[str, float], min_confidence_threshold: float=0.5) -> tuple[pd.Series, float]:
    """
    Finds the best matching row in the DataFrame for the given OCR text.
    """
    extracted_features = extract_features_from_ocr(ocr_text)
    print(f"Extracted Features for Matching: {extracted_features}") # Debugging

    # --- Optional: Pre-filtering based on reliable features ---
    candidates_df = df.copy()
    # Filter by ABV within a tolerance if found
    if extracted_features['abv'] is not None:
        tolerance = 1.0 # Allow +/- 1% ABV
        candidates_df = candidates_df[
            abs(candidates_df['abv'] - extracted_features['abv']) <= tolerance
        ]

    # Filter by Spirit Type if found
    if extracted_features['spirit_type_keywords']:
         # Match if the dataset spirit_type is *any* of the keywords found
         candidates_df = candidates_df[
             candidates_df['spirit_type'].isin(extracted_features['spirit_type_keywords'])
         ]

    # Filter by size if found
    if extracted_features['size_ml'] is not None:
        size_tolerance = 10 # Allow +/- 10ml
        candidates_df = candidates_df[
             abs(candidates_df['size_ml'] - extracted_features['size_ml']) <= size_tolerance
        ]

    print(f"Initial candidate count: {len(df)}, After filtering: {len(candidates_df)}") # Debugging

    if candidates_df.empty:
        print("No candidates found after filtering.")
        return None, 0.0 # No potential matches

    # --- Calculate scores for remaining candidates ---
    scores = candidates_df.apply(
        lambda row: calculate_similarity(extracted_features, row, weights),
        axis=1
    )

    if scores.empty:
        print("Scoring resulted in an empty series.")
        return None, 0.0

    # --- Find best match ---
    best_match_index = scores.idxmax()
    best_score = scores.max()

    # print(f"Scores:\n{scores}") # Debugging
    print(f"Best Score: {best_score:.4f} for Index: {best_match_index}") # Debugging

    if best_score >= min_confidence_threshold:
        best_match_row = df.loc[best_match_index]
        return best_match_row, best_score
    else:
        print(f"Best score {best_score:.4f} is below threshold {min_confidence_threshold}")
        return None, best_score # Return None if confidence is too low

