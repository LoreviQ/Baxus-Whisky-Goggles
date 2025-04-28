import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import pickle
from torch.utils.data import DataLoader

DEFAULTS = {"batch_size": 32, "num_workers": 4, "pin_memory": True, "shuffle": True}


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

    def train(self):
        train_loader = DataLoader(
            self.dataset,
            batch_size=self.config["batch_size"],
            shuffle=self.config["shuffle"],
            num_workers=self.config["num_workers"],
            pin_memory=self.config["pin_memory"],
        )

    def validate(self):
        val_loader = DataLoader(
            self.dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config["num_workers"],
            pin_memory=self.config["pin_memory"],
        )
