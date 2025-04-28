import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torchvision.models as models

import pickle
import torch
from torch.utils.data import DataLoader

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
        self._load_data(config)
        self._load_model()
        self._set_classifier()

    def _load_data(self, config):
        # Load the dataset
        initial_transform = transforms.Compose([])
        self.dataset = ImageFolder(root=self.dataset_path, transform=initial_transform)

        # Build loaders for training and validation
        self.train_loader = DataLoader(
            self.dataset,
            batch_size=config["batch_size"],
            shuffle=config["shuffle"],
            num_workers=config["num_workers"],
            pin_memory=config["pin_memory"],
        )
        self.val_loader = DataLoader(
            self.dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
            pin_memory=config["pin_memory"],
        )

    def _load_model(self):
        # Load the model
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        num_ftrs = model.classifier.in_features
        model.classifier = torch.nn.Linear(num_ftrs, NUM_CLASSES)

        # Freeze pre-trained layers, unfreeze the classifier
        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True

        # Move model to GPU if available
        self.model = model.to(
            torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        )

    def _set_classifier(self):
        # set loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # set optimizer
        params_to_update = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
        self.optimizer = torch.optim.AdamW(
            params_to_update, lr=0.001, weight_decay=0.01
        )

        # set learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=7, gamma=0.1
        )

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
