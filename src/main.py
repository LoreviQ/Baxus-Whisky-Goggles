import argparse
from utils.logger import setup_logger
from dataset.loader import DatasetLoader
from ocr import extract_text_from_image, preprocess_image
from PIL import Image, UnidentifiedImageError
import os
import numpy as np
import pandas as pd
from image.image_classification import ImageClassifier
from flask import Flask, request, jsonify
from io import BytesIO
import requests

dataset_path = os.path.join(os.path.dirname(__file__), "..", "data", "training_images")
pickle_path = os.path.join(os.path.dirname(__file__), "..", "data", "mapping_data.pkl")
model_path = os.path.join(
    os.path.dirname(__file__), "..", "models", "whiskey_goggles.pth"
)


def predict_image(
    image: Image.Image, loader: DatasetLoader, image_classifier: ImageClassifier, logger
):
    """
    Processes an image using OCR and Image Classification, combines results,
    and returns the best match.

    Args:
        image: The input PIL Image object.
        loader: An initialized DatasetLoader instance.
        image_classifier: An initialized ImageClassifier instance.
        logger: A configured logger instance.

    Returns:
        A pandas Series representing the best matching row, or None if no match found.
    """
    img_bytes = np.array(image)

    # OCR
    processed_image = preprocess_image(img_bytes, intensive=False)
    extracted_text = extract_text_from_image(processed_image)
    matches = loader.get_best_matches(extracted_text)
    logger.info(
        f"Extracted text: {extracted_text.strip().replace(chr(10), ' ')}"
    )  # Use chr(10) for newline
    logger.info(f"OCR Matches shape: {matches.shape}")

    # Image classification
    results = image_classifier.predict(image)
    logger.info(f"Image Classification Results shape: {results.shape}")

    # Ensure 'id' columns are numeric and compatible for merging
    logger.debug(
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

    logger.debug(
        f"Corrected dtypes - matches['id']: {matches['id'].dtype}, results['id']: {results['id'].dtype}"
    )

    # Join the results
    if matches.empty or results.empty:
        logger.warning(
            "No common IDs found between OCR matches and Image Classification results."
        )
        # Decide how to handle this: return empty, prioritize one, etc.
        # For now, returning None or an empty structure might be appropriate.
        # Option 1: Return None if no combined results
        # return None
        # Option 2: Return based on image classification if OCR fails or vice versa (needs more logic)
        # Option 3: Return an empty DataFrame/Series structure
        combined_results = pd.DataFrame(
            columns=["id", "name", "image_score", "ocr_score"]
        )  # Example empty structure
    else:
        combined_results = pd.merge(matches, results, on="id", how="inner")
        logger.info(f"Combined Results shape: {combined_results.shape}")
        if combined_results.empty:
            logger.warning("Inner merge resulted in an empty DataFrame. No common IDs.")

    # --- Combine Scores ---
    if not combined_results.empty:
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
                return pd.Series(
                    [1.0 if max_val > 0 else 0.0] * len(series), index=series.index
                )
            return (series - min_val) / (max_val - min_val)

        combined_results["image_score_norm"] = min_max_scale(
            combined_results["image_score"]
        )
        combined_results["ocr_score_norm"] = min_max_scale(
            combined_results["ocr_score_processed"]
        )

        # 3. Calculate Weighted Score
        combined_results["final_score"] = (
            image_weight * combined_results["image_score_norm"]
            + ocr_weight * combined_results["ocr_score_norm"]
        )

        # 4. Scale to Percentage
        combined_results["final_score_percent"] = combined_results["final_score"] * 100

        # 5. Sort by Final Score
        final_columns = [
            "id",
            "name",
            "final_score_percent",
            "image_score",
            "ocr_score",
        ]
        final_columns = [
            col for col in final_columns if col in combined_results.columns
        ]

        sorted_results = combined_results[final_columns].sort_values(
            by="final_score_percent", ascending=False
        )

        logger.info(f"Top results sorted by final score:\n{sorted_results.head()}")

        if not sorted_results.empty:
            return sorted_results.iloc[0]  # Return the best match (first row)
        else:
            logger.warning("Sorting resulted in an empty DataFrame.")
            return None
    else:
        logger.warning("Cannot combine scores as combined_results is empty.")
        return None
    # --- End Combine Scores ---


def main():
    # Set up app
    parser = argparse.ArgumentParser(description="Baxus Whisky Goggles API")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--host", default="127.0.0.1", help="Host to run the Flask app on"
    )
    parser.add_argument(
        "--port", type=int, default=5000, help="Port to run the Flask app on"
    )
    args = parser.parse_args()
    logger = setup_logger(args.verbose)
    logger.info("Starting application setup")

    # Initialize components
    try:
        loader = DatasetLoader()
        loader.process_ocr_data()  # Pre-load OCR data
        image_classifier = ImageClassifier(dataset_path, pickle_path, model_path)
        logger.info("DatasetLoader and ImageClassifier initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}", exc_info=True)
        return  # Exit if initialization fails

    # Set up Flask app
    app = Flask(__name__)

    @app.route("/predict", methods=["POST"])
    def handle_predict():
        logger.info("Received request for /predict")
        # Expect JSON payload with 'image_url'
        if not request.is_json:
            logger.warning("Request is not JSON.")
            return jsonify({"error": "Request must be JSON"}), 400

        data = request.get_json()
        if "image_url" not in data:
            logger.warning("No 'image_url' found in the request JSON.")
            return jsonify({"error": "'image_url' is required"}), 400

        image_url = data["image_url"]
        logger.info(f"Received image URL: {image_url}")

        try:
            # Fetch image from URL
            response = requests.get(image_url, stream=True, timeout=10)  # Add timeout
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

            # Check content type (optional but recommended)
            content_type = response.headers.get("content-type")
            if not content_type or not content_type.startswith("image/"):
                logger.warning(f"URL content type is not image: {content_type}")
                return jsonify({"error": "URL does not point to a valid image"}), 400

            # Read image content
            img_bytes = response.content
            image = Image.open(BytesIO(img_bytes)).convert("RGB")  # Ensure RGB
            logger.info(f"Image from URL '{image_url}' loaded successfully.")

            # Perform prediction
            best_match = predict_image(image, loader, image_classifier, logger)

            if best_match is not None and not best_match.empty:
                # Prepare response
                response_data = {
                    "id": int(best_match["id"]),  # Ensure id is standard int
                    "name": best_match["name"],
                    "final_score_percent": round(
                        best_match["final_score_percent"], 2
                    ),  # Round for cleaner output
                }
                logger.info(f"Prediction successful. Best match: {response_data}")
                return jsonify(response_data)
            else:
                logger.warning("Prediction returned no results.")
                return jsonify({"error": "Could not determine the best match"}), 404

        except requests.exceptions.RequestException as e:
            logger.error(
                f"Error fetching image from URL {image_url}: {e}", exc_info=True
            )
            return jsonify({"error": f"Could not fetch image from URL: {e}"}), 400
        except UnidentifiedImageError:  # Catch PIL errors for non-image data
            logger.error(
                f"Could not identify image from URL {image_url}. Content might not be an image.",
                exc_info=True,
            )
            return (
                jsonify(
                    {"error": "Failed to process image from URL. Invalid image format."}
                ),
                400,
            )
        except Exception as e:
            logger.error(
                f"Error processing image from URL {image_url}: {e}", exc_info=True
            )
            return jsonify({"error": "Internal server error during prediction"}), 500

    logger.info(f"Starting Flask server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port)  # Add host and port from args


if __name__ == "__main__":
    main()
