"""
Script to train the image classification model using PyTorch.
"""

import os
import argparse
from utils.logger import setup_logger
from image.image_classification import ImageClassifier, plot_accuracy

dataset_path = os.path.join(os.path.dirname(__file__), "..", "data", "training_images")
pickle_path = os.path.join(os.path.dirname(__file__), "..", "data", "mapping_data.pkl")
model_path = os.path.join(
    os.path.dirname(__file__), "..", "models", "whiskey_goggles.pth"
)


def train():
    image_classifier = ImageClassifier(dataset_path, pickle_path)
    image_classifier.train(50)
    plot_accuracy()


def validate():
    # Pass the trained model to the ImageClassifier
    image_classifier = ImageClassifier(dataset_path, pickle_path, model_path)
    image_classifier.validate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baxus Whisky Goggles")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    logger = setup_logger(args.verbose)
    logger.info("Starting application")
    validate()
