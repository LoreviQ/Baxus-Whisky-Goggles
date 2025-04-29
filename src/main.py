import argparse
from utils.logger import setup_logger
from dataset.loader import DatasetLoader
from ocr import extract_text_from_image, preprocess_image
from PIL import Image
import os
import numpy as np
import pandas as pd
from image.image_classification import ImageClassifier

dataset_path = os.path.join(os.path.dirname(__file__), "..", "data", "training_images")
pickle_path = os.path.join(os.path.dirname(__file__), "..", "data", "mapping_data.pkl")
model_path = os.path.join(
    os.path.dirname(__file__), "..", "models", "whiskey_goggles.pth"
)


def main():
    # Set up app
    parser = argparse.ArgumentParser(description="Baxus Whisky Goggles")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    logger = setup_logger(args.verbose)
    logger.info("Starting application")

    # Load dataset
    loader = DatasetLoader()
    loader.process_ocr_data()

    # Open test image
    image = Image.open(
        os.path.join(
            os.path.dirname(__file__), "..", "data", "augmented_images", "53003_1.png"
        )
    )
    img_bytes = np.array(image)

    # OCR
    processed_image = preprocess_image(img_bytes, intensive=False)
    extracted_text = extract_text_from_image(processed_image)
    matches = loader.get_best_matches(extracted_text)

    # Image classification
    image_classifier = ImageClassifier(dataset_path, pickle_path, model_path)
    results = image_classifier.predict(image)

    # Ensure 'id' columns are numeric and compatible for merging
    logger.info(
        f"Original dtypes - matches['id']: {matches['id'].dtype}, results['id']: {results['id'].dtype}"
    )
    matches["id"] = pd.to_numeric(matches["id"], errors="coerce")
    results["id"] = pd.to_numeric(results["id"], errors="coerce")

    # Drop rows where 'id' became NaN after coercion
    matches.dropna(subset=["id"], inplace=True)
    results.dropna(subset=["id"], inplace=True)

    # Cast to int64 if possible (no NaNs remaining)
    if not matches["id"].isnull().any():
        matches["id"] = matches["id"].astype("int64")
    if not results["id"].isnull().any():
        results["id"] = results["id"].astype("int64")

    logger.info(
        f"Corrected dtypes - matches['id']: {matches['id'].dtype}, results['id']: {results['id'].dtype}"
    )

    # Join the results
    combined_results = pd.merge(matches, results, on="id", how="inner")

    # --- Combine Scores ---
    ocr_threshold = 30
    image_weight = 0.7
    ocr_weight = 0.3

    # 1. Apply OCR Threshold
    combined_results["ocr_score_processed"] = combined_results["ocr_score"].apply(
        lambda x: x if x >= ocr_threshold else 0
    )

    # 2. Normalize Scores (Min-Max Scaling)
    def min_max_scale(series):
        min_val = series.min()
        max_val = series.max()
        if max_val - min_val == 0:
            # Handle case where all values are the same (avoid division by zero)
            # Return 1.0 for all if max_val > 0, else 0.0. Or adjust as needed.
            return pd.Series(
                [1.0 if max_val > 0 else 0.0] * len(series), index=series.index
            )
        return (series - min_val) / (max_val - min_val)

    if not combined_results.empty:
        combined_results["image_score_norm"] = min_max_scale(
            combined_results["image_score"]
        )
        combined_results["ocr_score_norm"] = min_max_scale(
            combined_results["ocr_score_processed"]
        )
    else:
        combined_results["image_score_norm"] = pd.Series(dtype="float64")
        combined_results["ocr_score_norm"] = pd.Series(dtype="float64")

    # 3. Calculate Weighted Score
    combined_results["final_score"] = (
        image_weight * combined_results["image_score_norm"]
        + ocr_weight * combined_results["ocr_score_norm"]
    )

    # 4. Scale to Percentage
    combined_results["final_score_percent"] = combined_results["final_score"] * 100

    # 5. Sort by Final Score
    # Select and sort columns
    final_columns = ["id", "name", "final_score_percent", "image_score", "ocr_score"]
    # Ensure columns exist before selecting
    final_columns = [col for col in final_columns if col in combined_results.columns]

    sorted_results = combined_results[final_columns].sort_values(
        by="final_score_percent", ascending=False
    )

    logger.info(f"Top results sorted by final score:\n{sorted_results.head()}")
    # --- End Combine Scores ---

    # logger.info(f"Sorted results by image score:\n{sorted_results}") # Keep or remove old log
    logger.info("Extracted text: %s", extracted_text.strip().replace("\n", "\\n"))
    logger.info(f"Shape of matches: {matches.shape}")
    logger.info(f"Shape of results: {results.shape}")
    logger.info(f"Shape of combined results: {combined_results.shape}")


if __name__ == "__main__":
    main()
