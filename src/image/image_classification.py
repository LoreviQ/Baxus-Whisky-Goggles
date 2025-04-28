import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import pickle

class ImageClassifier:
    def __init__(self, dataset_path, pickle_path):
        """
        Initialize the ImageClassifier with the dataset path and pickle path.
        """
        self.dataset_path = dataset_path
        self.pickle_path = pickle_path
        

    def save_class_mapping(self):
        """
        Save the class mapping from the dataset to a pickle file.
        """
        initial_transform = transforms.Compose([])
        full_dataset = ImageFolder(root=self.dataset_path, transform=initial_transform)
        class_to_idx = full_dataset.class_to_idx
        with open(self.pickle_path, 'wb') as f:
            pickle.dump(class_to_idx, f)

    def load_class_mapping(self):
        """
        Load the class mapping from a pickle file.
        """
        with open(self.pickle_path, 'rb') as f:
            class_to_idx = pickle.load(f)
        return class_to_idx