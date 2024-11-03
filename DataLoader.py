import os
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

##
## Define Hyperparameters and Paths
##
IMAGE_SIZE = 224  # Size to which each image will be resized (224x224 is standard for vision transformers)
BATCH_SIZE = 512  # Number of images per batch during training/testing
data_path = "/path/to/imagenet"  # Update this to the actual directory where ImageNet is stored

##
## Define Image Transformation Pipeline and DataLoader
##
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # Resize each image to IMAGE_SIZE x IMAGE_SIZE
    transforms.ToTensor(),  # Convert image to a PyTorch tensor (normalized to [0, 1])
])

##
## Custom ImageNet Dataset Class
##
class ImageNetKaggle(Dataset):
    """
    Custom Dataset class for loading ImageNet images and their corresponding labels.

    This class supports training, validation, and testing splits. It assumes the ImageNet directory has:
    - A subdirectory for each split ('train', 'val', and 'test'), containing images organized by class for training.
    - JSON files to map synset IDs to class IDs for validation and testing splits.

    Attributes:
    - root (str): Base directory where ImageNet data is stored.
    - split (str): Data split to load ('train', 'val', or 'test').
    - transform: Transformations applied to each image.
    - samples (list): List of image file paths.
    - targets (list): Corresponding labels for each image.
    - syn_to_class (dict): Mapping from synset IDs to integer class labels.
    - val_to_syn (dict): Mapping from validation image file names to synset IDs.
    """
    def __init__(self, root, split='train', transform=None):
        self.samples = []       # List to store file paths for each image in the dataset
        self.targets = []       # List to store corresponding class labels for each image
        self.transform = transform  # Transformations applied to each image
        self.syn_to_class = {}  # Mapping from synset ID (e.g., 'n01440764') to integer class label (0, 1, ...)

        # Load class index JSON for mapping synset to class ID (required for validation and testing)
        with open(os.path.join(root, "imagenet_class_index.json"), "r") as f:
            json_file = json.load(f)
            # Populate syn_to_class dictionary, which maps synset IDs to class IDs
            for class_id, v in json_file.items():
                self.syn_to_class[v[0]] = int(class_id)

        # For validation and testing, load additional JSON mapping validation images to synset IDs
        if split == "val" or split == "test":
            with open(os.path.join(root, "ILSVRC2012_val_labels.json"), "r") as f:
                self.val_to_syn = json.load(f)

        # Directory for the specific data split (train/val/test) under the ILSVRC dataset directory
        samples_dir = os.path.join(root, "ILSVRC/Data/CLS-LOC", split)

        # Populate samples and targets lists based on the data split
        for entry in os.listdir(samples_dir):
            if split == "train":
                # In training, each class has its own folder, identified by synset ID
                syn_id = entry  # Folder name is the synset ID
                target = self.syn_to_class.get(syn_id, None)
                if target is not None:
                    syn_folder = os.path.join(samples_dir, syn_id)  # Path to images for this synset ID
                    for sample in os.listdir(syn_folder):
                        sample_path = os.path.join(syn_folder, sample)
                        self.samples.append(sample_path)  # Add image path to samples
                        self.targets.append(target)       # Add corresponding target label to targets
            elif split == "val" or split == "test":
                # For validation and testing, images are directly in the folder and need mapping to class labels
                syn_id = self.val_to_syn.get(entry, None)  # Look up synset ID using file name
                target = self.syn_to_class.get(syn_id, None)
                if target is not None:
                    sample_path = os.path.join(samples_dir, entry)
                    self.samples.append(sample_path)  # Add image path to samples
                    self.targets.append(target)       # Add corresponding target label to targets

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
## Instantiate Datasets and DataLoaders for Train, Validation, and Test
##

# Instantiate the training dataset and DataLoader
train_dataset = ImageNetKaggle(root=data_path, split='train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Instantiate the validation dataset and DataLoader
val_dataset = ImageNetKaggle(root=data_path, split='val', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Instantiate the test dataset and DataLoader
test_dataset = ImageNetKaggle(root=data_path, split='test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Example usage: Iterating over the train_loader
# This is just to check if the dataset loads correctly and will be removed during actual training
for images, labels in train_loader:
    print(f"Batch of images has shape: {images.shape}")
    print(f"Batch of labels has shape: {labels.shape}")
    break
