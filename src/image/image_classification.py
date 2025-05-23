import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torchvision.models as models

import pickle
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

import os
import time
import copy
import logging
import warnings
import csv
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


# --- Configuration ---
DEFAULTS = {
    "batch_size": 32,
    "num_workers": 4,
    "pin_memory": True,
    "shuffle": True,
    "fine_tune_blocks": 1,  # Number of blocks to fine-tune (head_only = 0)
    "head_only_lr": 3e-4,
    "ft_head_lr": 1e-4,
    "ft_feature_lr": 2e-5,
}
NUM_CLASSES = 500
MODEL_NAME = "efficientnet_b0"  # Keep B0 for now

# --- Setup ---
logger = logging.getLogger("BaxusLogger")
warnings.filterwarnings("ignore", category=UserWarning, module="PIL.TiffImagePlugin")
warnings.filterwarnings("ignore", category=UserWarning, module="PIL.Image")


class ImageClassifier:
    def __init__(self, dataset_path, pickle_path, model_path=None, config=DEFAULTS):
        """
        Initialize the ImageClassifier with the dataset path and pickle path.
        """
        self.dataset_path = dataset_path
        self.pickle_path = pickle_path
        self._load_data(config)
        self._load_model()
        self._set_classifier(config)
        if model_path:
            self._load_trained_model(model_path)

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
                transforms.Lambda(lambda img: img.convert("RGB")),
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
        self.val_transforms = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.Lambda(lambda img: img.convert("RGB")),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        val_dataset = ImageFolder(root=self.dataset_path, transform=self.val_transforms)
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

    def _set_classifier(self, config):
        # Set loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # --- Determine Parameters and Learning Rates ---
        trainable_parameter_groups = []
        head_params = list(self.model.classifier.parameters())

        if config["fine_tune_blocks"] > 0:
            logger.info(
                f"Setting up optimizer for fine-tuning: Unfreezing last {config['fine_tune_blocks']} feature block(s)."
            )
            # Find the feature blocks to unfreeze
            try:
                # EfficientNet features are typically in model.features
                num_feature_blocks = len(self.model.features)
                if config["fine_tune_blocks"] > num_feature_blocks:
                    logger.warning(
                        f"Requested fine_tune_blocks ({config['fine_tune_blocks']}) > available blocks ({num_feature_blocks}). Unfreezing all feature blocks."
                    )
                    config["fine_tune_blocks"] = num_feature_blocks

                # Slice the feature blocks from the end
                blocks_to_unfreeze = self.model.features[-config["fine_tune_blocks"] :]
                feature_params_to_update = []
                for block in blocks_to_unfreeze:
                    for param in block.parameters():
                        param.requires_grad = True
                        feature_params_to_update.append(param)

                logger.info(
                    f"Unfroze {len(blocks_to_unfreeze)} feature block(s) containing {len(feature_params_to_update)} parameters."
                )

                # Setup parameter groups for differential LRs
                trainable_parameter_groups = [
                    {
                        "params": feature_params_to_update,
                        "lr": config["ft_feature_lr"],
                    },
                    {
                        "params": head_params,
                        "lr": config["ft_head_lr"],
                    },
                ]
                current_head_lr = config["ft_head_lr"]
                logger.info(
                    f"Optimizer setup: Feature LR={config['ft_feature_lr'],}, Head LR={config['ft_head_lr']}"
                )

            except AttributeError:
                logger.error(
                    "Could not access model.features. Fine-tuning setup failed. Defaulting to head-only."
                )
                config["fine_tune_blocks"] = 0
                trainable_parameter_groups = [{"params": head_params}]
                current_head_lr = config["head_only_lr"]
                logger.info(f"Optimizer setup: Head-only LR={current_head_lr}")

        else:
            logger.info("Setting up optimizer for head-only training.")
            trainable_parameter_groups = [{"params": head_params}]
            current_head_lr = config["head_only_lr"]
            logger.info(f"Optimizer setup: Head-only LR={current_head_lr}")

        # Set optimizer using the configured parameter groups
        self.optimizer = torch.optim.AdamW(
            trainable_parameter_groups,
            lr=current_head_lr,
            weight_decay=0.01,
        )

        # --- Set Learning Rate Scheduler ---
        # Use ReduceLROnPlateau when fine-tuning, StepLR otherwise (example logic)
        if config["fine_tune_blocks"] > 0:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.2, patience=5, verbose=True
            )
            logger.info("Using ReduceLROnPlateau scheduler monitoring validation loss.")
        else:
            # Keep StepLR for head-only, maybe make it less aggressive?
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=10, gamma=0.5
            )  # Example: Slower decay
            logger.info(
                f"Using StepLR scheduler: step={self.scheduler.step_size}, gamma={self.scheduler.gamma}"
            )

        # Log final list of trainable parameters for verification
        logger.info("Final list of trainable parameters:")
        total_trainable_params = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                logger.info(f"\t{name} (Shape: {param.shape})")
                total_trainable_params += param.numel()
        logger.info(f"Total Trainable Parameters: {total_trainable_params:,}")

    def _load_trained_model(self, model_path):
        if not hasattr(self, "model"):
            logger.error("Model architecture not initialized. Cannot load weights.")
            return False
        if not hasattr(self, "device"):
            logger.error("Device not set. Cannot load weights.")
            return False
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            logger.info(
                f"Successfully loaded trained model weights from {model_path} and set to eval mode."
            )
            return True
        except FileNotFoundError:
            logger.error(f"Model file not found: {model_path}")
            return False
        except Exception as e:
            logger.error(f"Failed to load model weights from {model_path}: {e}")
            return False

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
        Train the model for the specified number of epochs and save accuracy history.
        """
        since = time.time()
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        model_save_dir = "models"
        os.makedirs(model_save_dir, exist_ok=True)
        accuracy_log_path = os.path.join(model_save_dir, "accuracy_log.csv")

        # Initialize CSV log file
        try:
            with open(accuracy_log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["epoch", "train_loss", "train_acc", "val_loss", "val_acc"]
                )
            logger.info(f"Initialized accuracy log at {accuracy_log_path}")
        except IOError as e:
            logger.error(f"Error initializing accuracy log file: {e}")
            return self.model, None

        # Training loop
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
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total_samples += inputs.size(0)

            epoch_train_loss = running_loss / total_samples
            epoch_train_acc = running_corrects.double() / total_samples
            current_lr = self.optimizer.param_groups[0]["lr"]
            logger.info(
                f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} LR: {current_lr:.6f}"
            )

            # --- Validation Phase ---
            self.model.eval()
            running_loss = 0.0
            running_corrects = 0
            total_samples = 0
            with torch.no_grad():
                for inputs, labels in self.val_loader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = self.criterion(outputs, labels)
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    total_samples += inputs.size(0)

            epoch_val_loss = running_loss / total_samples
            epoch_val_acc = running_corrects.double() / total_samples
            logger.info(f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")

            # --- Log Accuracy to CSV ---
            try:
                with open(accuracy_log_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            epoch,
                            epoch_train_loss,
                            epoch_train_acc.item(),
                            epoch_val_loss,
                            epoch_val_acc.item(),
                        ]
                    )
            except IOError as e:
                logger.error(
                    f"Error writing to accuracy log file at epoch {epoch}: {e}"
                )

            # --- Save Best Model ---
            if epoch_val_acc > best_acc:
                best_acc = epoch_val_acc
                best_model_wts = copy.deepcopy(self.model.state_dict())
                save_filename = f"whiskey_goggles_{MODEL_NAME}_best_epoch{epoch}.pth"
                save_path = os.path.join(model_save_dir, save_filename)
                torch.save(self.model.state_dict(), save_path)
                logger.info(
                    f"Saved new best model (epoch {epoch}) with validation accuracy: {best_acc:.4f} to {save_path}"
                )

            # --- Update Learning Rate ---
            if self.scheduler:
                if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step(epoch_val_loss)
                else:
                    self.scheduler.step()

        time_elapsed = time.time() - since
        logger.info(
            f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
        )
        logger.info(f"Best val Acc: {best_acc:4f}")

        # Load best model weights
        self.model.load_state_dict(best_model_wts)
        # Return model and path to accuracy log instead of history dict
        return self.model, accuracy_log_path

    def validate(self, results_dir="results"):
        """
        Validate the model on the validation dataset.
        """
        logger.info("Starting model validation...")
        if not hasattr(self, "model") or not hasattr(self, "val_loader"):
            logger.error("Model or validation loader not initialized. Cannot validate.")
            return

        # Ensure results directory exists
        os.makedirs(results_dir, exist_ok=True)

        # Load class mapping
        _, idx_to_class = self.load_class_mapping()
        if not idx_to_class:
            logger.error(
                "Failed to load class mapping. Cannot proceed with validation."
            )
            return
        class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

        # Set model to evaluation mode
        self.model.eval()

        all_preds = []
        all_labels = []

        logger.info("Running predictions on validation set...")
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        if not all_labels or not all_preds:
            logger.error("No predictions were generated from the validation set.")
            return

        logger.info(f"Generated {len(all_preds)} predictions.")

        # --- Calculate and Print Metrics ---
        logger.info("\n--- Classification Report ---")
        try:
            # Use target_names if few classes, otherwise None
            target_names_display = class_names if len(class_names) <= 50 else None
            report = classification_report(
                all_labels,
                all_preds,
                target_names=target_names_display,
                digits=3,
                zero_division=0,
            )
            print(report)
            logger.info("Classification report generated.")
            report_path = os.path.join(results_dir, "classification_report.txt")
            with open(report_path, "w") as f:
                f.write(report)
            logger.info(f"Classification report saved to {report_path}")

        except Exception as e:
            logger.error(f"Error generating classification report: {e}")

        # --- Generate and Save Confusion Matrix ---
        logger.info("\n--- Generating Confusion Matrix ---")
        try:
            cm = confusion_matrix(
                all_labels, all_preds, labels=list(range(len(class_names)))
            )

            # Plotting (may be large for 500 classes)
            display_labels = class_names if len(class_names) <= 50 else None
            fig, ax = plt.subplots(figsize=(20, 20))
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm, display_labels=display_labels
            )
            disp.plot(
                cmap=plt.cm.Blues, ax=ax, xticks_rotation="vertical", values_format="d"
            )
            plt.title("Confusion Matrix")
            plt.tight_layout()
            cm_path = os.path.join(results_dir, "confusion_matrix.png")
            plt.savefig(cm_path)
            plt.close(fig)
            logger.info(f"Confusion matrix plot saved to {cm_path}")
        except Exception as e:
            logger.error(f"Could not generate or save confusion matrix plot: {e}")
        logger.info("Validation finished.")

    def predict(self, image):
        """
        Identifies the bottle in a single image.

        Args:
            image (np.ndarray): input image file.
            top_k (int): Number of top predictions to return.

        Returns:
            list: A list of dictionaries [{'bottle_id': str, 'score': float},...],
                  or None if an error occurs.
        """
        if not hasattr(self, "model") or not hasattr(self, "val_transforms"):
            logger.error("Model or transforms not initialized. Cannot predict.")
            return None

        # Load class mapping (needed to translate index to bottle ID)
        class_to_idx, idx_to_class = self.load_class_mapping()
        if not idx_to_class:
            logger.error("Failed to load class mapping. Cannot identify.")
            return None
        num_classes = len(class_to_idx)
        self.model.eval()

        try:
            # 1. Load and Preprocess Image
            img_tensor = self.val_transforms(image)
            img_tensor = img_tensor.unsqueeze(0)
            img_tensor = img_tensor.to(self.device)

            # 2. Make Prediction
            with torch.no_grad():
                outputs = self.model(img_tensor)

            # 3. Get Probabilities and Rank
            probabilities = F.softmax(outputs, dim=1)
            scores = probabilities.squeeze(0).cpu().tolist()

            # 4. Format Results
            bottle_ids = [
                idx_to_class.get(i, f"Unknown Index {i}") for i in range(num_classes)
            ]
            results_df = pd.DataFrame(
                {
                    "id": bottle_ids,  # Use 'id' to match your OCR DataFrame column name
                    "image_score": scores,
                }
            )
            logger.info(f"Generated scores for all {num_classes} classes")
            return results_df
        except Exception as e:
            logger.error(f"Error identifying image: {e}")
            return None


# --- Plotting Function (Outside the class) ---
def plot_accuracy(csv_path="models/accuracy_log.csv", save_plot=True):
    """
    Loads accuracy data from a CSV file and plots training vs validation accuracy.

    Args:
        csv_path (str): Path to the CSV file containing accuracy data.
        save_plot (bool): Whether to save the plot as an image file.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Accuracy log file not found at {csv_path}")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(df["epoch"], df["train_acc"], label="Training Accuracy")
    plt.plot(df["epoch"], df["val_acc"], label="Validation Accuracy")

    plt.title("Training and Validation Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    if save_plot:
        plot_filename = os.path.splitext(csv_path)[0] + ".png"
        try:
            plt.savefig(plot_filename)
            print(f"Plot saved to {plot_filename}")
        except Exception as e:
            print(f"Error saving plot: {e}")
    else:
        plt.show()
