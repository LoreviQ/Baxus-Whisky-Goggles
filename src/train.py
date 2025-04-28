"""
Script to train the image classification model using PyTorch.
"""

import os
from image.image_classification import ImageClassifier

dataset_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'training_images')
pickle_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'class_to_idx.pkl')

if __name__ == "__main__":
    image_classifier = ImageClassifier(dataset_path, pickle_path)