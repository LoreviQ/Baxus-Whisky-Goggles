import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torchvision.models as models

import pickle
import torch
from torch.utils.data import DataLoader

import time
import copy
import logging

DEFAULTS = {"batch_size": 32, "num_workers": 4, "pin_memory": True, "shuffle": True}
NUM_CLASSES = 500
MODEL_NAME = "efficientnet_b0"
logger = logging.getLogger("BaxusLogger")


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
        # Training data
        train_transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomResizedCrop(size=(224, 224), scale=(0.6, 1.0)),
                transforms.RandomRotation(15),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
            ]
        )
        train_dataset = ImageFolder(root=self.dataset_path, transform=train_transforms)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=config["shuffle"],
            num_workers=config["num_workers"],
            pin_memory=config["pin_memory"],
        )
        logger.info(
            f"Created Train DataLoader with {len(train_dataset)} samples and augmentations."
        )

        # Validation data
        val_transforms = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        val_dataset = ImageFolder(root=self.dataset_path, transform=val_transforms)
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
            pin_memory=config["pin_memory"],
        )
        logger.info(
            f"Created Validation DataLoader with {len(val_dataset)} samples and minimal transforms."
        )

        # Save class mapping
        class_to_idx = train_dataset.class_to_idx
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        mapping_data = {"class_to_idx": class_to_idx, "idx_to_class": idx_to_class}
        with open(self.pickle_path, "wb") as f:
            pickle.dump(mapping_data, f)
        logger.info(f"Saved class mapping to {self.pickle_path}")

    def _load_model(self):
        # Load the model
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(num_ftrs, NUM_CLASSES)

        # Freeze pre-trained layers, unfreeze the classifier
        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier[-1].parameters():
            param.requires_grad = True

        # Move model to GPU if available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        # Logging
        logger.info(
            f"Loaded EfficientNet-B0, replaced final layer for {NUM_CLASSES} classes."
        )
        logger.info(f"Model moved to device: {self.device}")
        logger.info("Parameters to train:")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                logger.info(f"\t{name}")

    def _set_classifier(self):
        # set loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # set optimizer
        params_to_update = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
        self.optimizer = torch.optim.AdamW(
            params_to_update, lr=0.0001, weight_decay=0.01
        )

        # set learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=7, gamma=0.1
        )

    def load_class_mapping(self):
        """
        Load the class mapping from a pickle file.
        """
        try:
            with open(self.pickle_path, "rb") as f:
                mapping_data = pickle.load(f)
            loaded_class_to_idx = mapping_data.get("class_to_idx")
            loaded_idx_to_class = mapping_data.get("idx_to_class")
            return loaded_class_to_idx, loaded_idx_to_class
        except Exception as e:
            print(f"Error loading mappings: {e}")

    def train(self, num_epochs=25):
        """
        Train the model for the specified number of epochs.
        """
        since = time.time()
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch}/{num_epochs - 1}")
            logger.info("-" * 10)

            # --- Training Phase ---
            self.model.train()
            running_loss = 0.0
            running_corrects = 0
            total_samples = 0

            for inputs, labels in self.train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)

                # Backward pass and optimize (only updates params with requires_grad=True)
                loss.backward()
                self.optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total_samples += inputs.size(0)

            epoch_loss = running_loss / total_samples
            epoch_acc = running_corrects.double() / total_samples
            history["train_loss"].append(epoch_loss)
            history["train_acc"].append(epoch_acc.item())  # Convert tensor to float
            logger.info(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # --- Validation Phase ---
            self.model.eval()  # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0
            total_samples = 0

            # Use the validation loader (applies val_transform)
            with torch.no_grad():  # Disable gradient calculation
                for inputs, labels in self.val_loader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = self.criterion(outputs, labels)

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    total_samples += inputs.size(0)

            epoch_loss = running_loss / total_samples
            epoch_acc = running_corrects.double() / total_samples
            history["val_loss"].append(epoch_loss)
            history["val_acc"].append(epoch_acc.item())
            logger.info(f"Val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # --- Save Best Model ---
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(self.model.state_dict())
                # Save the best model checkpoint
                torch.save(
                    self.model.state_dict(), f"best_model_{MODEL_NAME}_epoch{epoch}.pth"
                )
                logger.info(f"Saved new best model with accuracy: {best_acc:.4f}")

            # --- Update Learning Rate ---
            if self.scheduler:
                self.scheduler.step()

        time_elapsed = time.time() - since
        logger.info(
            f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
        )
        logger.info(f"Best val Acc: {best_acc:4f}")

        # Load best model weights
        self.model.load_state_dict(best_model_wts)
        return self.model, history
