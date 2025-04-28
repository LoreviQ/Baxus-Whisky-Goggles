"""
Script to train the image classification model using PyTorch.
"""

import os
from image.image_classification import save_class_mapping

dataset_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'training_images')
pickle_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'class_to_idx.pkl')

if __name__ == "__main__":
    save_class_mapping(dataset_path, pickle_path)