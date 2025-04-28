import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import pickle
from torch.utils.data import DataLoader
import torch
import torchvision.models as models
import torch.nn as nn

DEFAULTS = {"batch_size": 32, "num_workers": 4, "pin_memory": True, "shuffle": True}
NUM_CLASSES = 500
MODEL_NAME = "efficientnet_b0"


class ImageClassifier:
    def __init__(self, dataset_path, pickle_path, config=DEFAULTS):
        """
        Initialize the ImageClassifier with the dataset path and pickle path.
        """
        self.dataset_path = dataset_path
        self.pickle_path = pickle_path
        self.config = config
        self.dataset = self._load_data()

    def _load_data(self):
        initial_transform = transforms.Compose([])
        return ImageFolder(root=self.dataset_path, transform=initial_transform)

    def save_class_mapping(self):
        """
        Save the class mapping from the dataset to a pickle file.
        """

        class_to_idx = self.dataset.class_to_idx
        with open(self.pickle_path, "wb") as f:
            pickle.dump(class_to_idx, f)

    def load_class_mapping(self):
        """
        Load the class mapping from a pickle file.
        """
        with open(self.pickle_path, "rb") as f:
            class_to_idx = pickle.load(f)
        return class_to_idx

    def augment_data(self):
        """
        Apply data augmentation to the dataset.
        """
        augment_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            ]
        )
        self.dataset.transform = augment_transform

    def train(self):
        # Load the dataset
        train_loader = DataLoader(
            self.dataset,
            batch_size=self.config["batch_size"],
            shuffle=self.config["shuffle"],
            num_workers=self.config["num_workers"],
            pin_memory=self.config["pin_memory"],
        )

        # Load the model
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, NUM_CLASSES)

    def validate(self):
        val_loader = DataLoader(
            self.dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config["num_workers"],
            pin_memory=self.config["pin_memory"],
        )
