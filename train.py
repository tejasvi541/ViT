import os
import timeit
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from DataLoader import train_loader, val_loader, test_loader  # Import the dataloaders
from ViT import ViT  # Import the ViT model
import numpy as np

# Set random seed for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)

# Hyperparameters
LEARNING_RATE = 0.0001
BATCH_SIZE = 128
EPOCHS = 50
NUM_CLASSES = 257
PATCH_SIZE = 16
IMAGE_SIZE = 224
IN_CHANNELS = 3
NUM_HEADS = 8
DROPOUT = 0.001
HIDDEN_DIM = 768
ADAM_WEIGHT_DECAY = 0
ADAM_BETAS = (0.9, 0.999)
ACTIVATION = "gelu"
NUM_ENCODER = 4
EMBED_DIM = (PATCH_SIZE**2) * IN_CHANNELS  # For the ViT model
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2  # Number of patches in the image

# Device configuration (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the ViT model
model = ViT(
    num_patches=NUM_PATCHES,
    num_classes=NUM_CLASSES,
    patch_size=PATCH_SIZE,
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_ENCODER,
    dropout=DROPOUT,
    activation=ACTIVATION,
    in_channels=IN_CHANNELS,
).to(device)

# Optimizer and Loss Function
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS, weight_decay=ADAM_WEIGHT_DECAY)
criterion = nn.CrossEntropyLoss()

# Function to train the model for one epoch
def train_one_epoch(model, train_loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.

    Parameters:
    - model: The ViT model being trained.
    - train_loader: DataLoader for training data.
    - optimizer: Optimizer for updating model parameters.
    - criterion: Loss function (CrossEntropyLoss).
    - device: Device to use for computation ('cuda' or 'cpu').

    Returns:
    - average_train_loss: Average loss across the epoch.
    - train_accuracy: Accuracy across the training set.
    """
    model.train()  # Set model to training mode
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    # Loop through batches in the training data
    for images, labels in tqdm(train_loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        # Forward pass: Compute model output and calculate loss
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Zero gradients, backward pass, and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track the loss and accuracy
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)

    # Calculate average loss and accuracy
    average_train_loss = running_loss / total_samples
    train_accuracy = correct_predictions / total_samples

    return average_train_loss, train_accuracy


# Function to validate the model
def validate(model, val_loader, criterion, device):
    """
    Validates the model on the validation dataset.

    Parameters:
    - model: The ViT model being validated.
    - val_loader: DataLoader for validation data.
    - criterion: Loss function (CrossEntropyLoss).
    - device: Device to use for computation ('cuda' or 'cpu').

    Returns:
    - average_val_loss: Average loss across the validation set.
    - val_accuracy: Accuracy across the validation set.
    """
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():  # Disable gradient calculation
        for images, labels in tqdm(val_loader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)

            # Forward pass and loss calculation
            outputs = model(images)
            loss = criterion(outputs, labels)

            print("Target labels:", labels)  # Debugging line
            print("Model outputs:", outputs)

            # Track the loss and accuracy
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    # Calculate average loss and accuracy
    average_val_loss = running_loss / total_samples
    val_accuracy = correct_predictions / total_samples

    return average_val_loss, val_accuracy


# Training loop for multiple epochs
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(1, EPOCHS + 1):
    print(f"\nEpoch {epoch}/{EPOCHS}")

    # Train and validate for this epoch
    train_loss, train_accuracy = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_accuracy = validate(model, val_loader, criterion, device)

    # Append results for analysis
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)

    # Feedback: Display epoch summary
    print(f"Epoch [{epoch}/{EPOCHS}] - "
          f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4%} - "
          f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4%}")

    # Improved feedback: Save checkpoint if validation improves
    if epoch == 1 or val_accuracy > max(val_accuracies[:-1]):
        torch.save(model.state_dict(), f"best_model_epoch_{epoch}.pth")
        print(f"Model saved for epoch {epoch} with val accuracy: {val_accuracy:.4%}")

# Save final model
torch.save(model.state_dict(), "final_model.pth")
print("Training complete. Final model saved.")
