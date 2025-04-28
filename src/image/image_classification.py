import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import pickle

def save_class_mapping(dataset_path, pickle_path):
    """
    Save the class mapping from the dataset to a pickle file.
    """
    initial_transform = transforms.Compose([])
    full_dataset = ImageFolder(root=dataset_path, transform=initial_transform)
    class_to_idx = full_dataset.class_to_idx
    with open(pickle_path, 'wb') as f:
        pickle.dump(class_to_idx, f)