import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.datasets.utils import download_and_extract_archive
import glob

##
## Define Hyperparameters and Paths
##
IMAGE_SIZE = 224  # Resize each image to 224x224, typical for vision transformers
BATCH_SIZE = 512  # Number of images per batch during training/testing
data_path = "./data/caltech256"  # Path where Caltech-256 data will be stored

# URL for downloading the Caltech-256 dataset archive
URL = "https://data.caltech.edu/records/nyy15-4j048/files/256_ObjectCategories.tar"

##
## Download and Extract Caltech-256 Dataset
##
def download_caltech256(root, url=URL):
    """
    Download and extract the Caltech-256 dataset if it's not already present.

    Parameters:
    - root (str): Directory where the dataset will be stored
    - url (str): URL of the dataset archive
    """
    # Check if dataset directory exists
    if not os.path.exists(os.path.join(root, "256_ObjectCategories")):
        os.makedirs(root, exist_ok=True)
        # Download and extract dataset
        download_and_extract_archive(url, download_root=root)
        print("Caltech-256 dataset downloaded and extracted successfully.")
    else:
        print("Caltech-256 dataset already exists.")

# Download dataset if not already present
download_caltech256(data_path)

##
## Define Image Transformation Pipeline and DataLoader
##
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # Resize each image to IMAGE_SIZE x IMAGE_SIZE
    transforms.ToTensor(),  # Convert image to a PyTorch tensor (normalized to [0, 1])
])

##
## Custom Caltech-256 Dataset Class
##
class Caltech256Dataset(Dataset):
    """
    Custom Dataset class for loading Caltech-256 images and their corresponding labels.

    This class assumes the Caltech-256 directory has images organized in folders named by class IDs (e.g., '001.ak47').

    Attributes:
    - root (str): Base directory where Caltech-256 data is stored.
    - transform: Transformations applied to each image.
    - samples (list): List of image file paths.
    - targets (list): Corresponding labels for each image.
    - class_to_idx (dict): Mapping from class names to integer labels.
    """
    def __init__(self, root, transform=None):
        self.samples = []       # List to store file paths for each image in the dataset
        self.targets = []       # List to store corresponding class labels for each image
        self.transform = transform  # Transformations applied to each image
        self.class_to_idx = {}  # Mapping from class folder names to integer class labels

        # Check if root directory exists
        if not os.path.isdir(root):
            raise FileNotFoundError(f"Directory not found: {root}")

        # Populate class_to_idx dictionary and samples/targets lists
        for idx, class_name in enumerate(sorted(os.listdir(root))):
            class_dir = os.path.join(root, class_name)
            if os.path.isdir(class_dir):
                self.class_to_idx[class_name] = idx  # Map class name to index
                for img_path in glob.glob(os.path.join(class_dir, "*.jpg")):
                    self.samples.append(img_path)  # Add image path to samples
                    self.targets.append(idx)       # Add corresponding target label to targets

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Fetch the image and label at the specified index.

        Parameters:
        - idx: Index of the data point to retrieve

        Returns:
        - Transformed image tensor and its corresponding class label
        """
        # Load image file and convert it to RGB
        x = Image.open(self.samples[idx]).convert("RGB")
        
        # Apply transformations (e.g., resizing, tensor conversion)
        if self.transform:
            x = self.transform(x)
            
        # Return the transformed image and its label
        return x, self.targets[idx]

##
## Instantiate Dataset and DataLoader
##

# Instantiate the full dataset
dataset = Caltech256Dataset(root=os.path.join(data_path, "256_ObjectCategories"), transform=transform)

# Split the dataset into train (80%), validation (10%), and test (10%) sets
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create DataLoaders for each split
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Example usage: Iterating over the train_loader
# This is just to check if the dataset loads correctly and will be removed during actual training
for images, labels in train_loader:
    print(f"Batch of images has shape: {images.shape}")
    print(f"Batch of labels has shape: {labels.shape}")
    break
