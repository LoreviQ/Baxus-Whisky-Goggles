"""
Script to train the image classification model using PyTorch.
"""

import os
import argparse
from utils.logger import setup_logger
from image.image_classification import ImageClassifier, plot_accuracy
from PIL import Image

dataset_path = os.path.join(os.path.dirname(__file__), "..", "data", "training_images")
pickle_path = os.path.join(os.path.dirname(__file__), "..", "data", "mapping_data.pkl")
model_path = os.path.join(
    os.path.dirname(__file__), "..", "models", "whiskey_goggles.pth"
)
# For testing
predict_path = os.path.join(os.path.dirname(__file__), "..", "data", "augmented_images")


def train():
    image_classifier = ImageClassifier(dataset_path, pickle_path)
    image_classifier.train(50)
    plot_accuracy()


def validate():
    # Pass the trained model to the ImageClassifier
    image_classifier = ImageClassifier(dataset_path, pickle_path, model_path)
    image_classifier.validate()


def predict(filename):
    # Pass the trained model to the ImageClassifier
    image_classifier = ImageClassifier(dataset_path, pickle_path, model_path)
    image = Image.open(os.path.join(predict_path, filename))
    results = image_classifier.predict(image)
    print(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baxus Whisky Goggles")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--validate", action="store_true", help="Validate the model")
    parser.add_argument(
        "--predict", action="store_true", help="Predict using the model"
    )
    args = parser.parse_args()

    logger = setup_logger(args.verbose)
    logger.info("Starting application")
    if args.validate:
        logger.info("Validating the model")
        validate()
    elif args.predict:

        predict("53003_1.png")
    else:
        logger.info("Training the model")
        train()
