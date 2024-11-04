import os
import timeit
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from DataLoader import train_dataloader, val_dataloader  # Import the dataloaders
from ViT import ViT  # Import the ViT model

# Set random seed for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)

# Hyperparameters
LEARNING_RATE = 0.0001
BATCH_SIZE = 512
EPOCHS = 50
NUM_CLASSES = 1000
PATCH_SIZE = 16
IMAGE_SIZE = 224
IN_CHANNELS = 3
NUM_HEADS = 8
DROPOUT = 0.001
HIDDEN_DIM = 768
ADAM_WEIGHT_DECAY = 0
ADAM_BETAS = (0.9, 0.999)
ACTIVATION = "gelu"
NUM_ENCODER = 2
EMBED_DIM = (PATCH_SIZE**2) * IN_CHANNELS  # For the ViT model
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2  # Number of patches in the image

# Device configuration (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the ViT model
model = ViT(num_classes=NUM_CLASSES, patch_size=PATCH_SIZE, in_channels=IN_CHANNELS,
            num_heads=NUM_HEADS, hidden_dim=HIDDEN_DIM, dropout=DROPOUT,
            activation=ACTIVATION, num_encoder=NUM_ENCODER, embed_dim=EMBED_DIM).to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Loss function for multi-class classification
optimizer = optim.Adam(model.parameters(), betas=ADAM_BETAS, lr=LEARNING_RATE, weight_decay=ADAM_WEIGHT_DECAY)

# Start timing the training process
start = timeit.default_timer()

# Training loop
for epoch in tqdm(range(EPOCHS), position=0, leave=True):
    model.train()  # Set the model to training mode
    train_labels = []
    train_preds = []
    train_running_loss = 0
    
    # Iterate over training batches
    for idx, img_label in enumerate(tqdm(train_dataloader, position=0, leave=True)):
        img = img_label["image"].float().to(device)  # Move image to device (GPU/CPU)
        label = img_label["label"].type(torch.long).to(device)  # Move label to device
        
        # Forward pass
        y_pred = model(img)  # Get predictions from the model
        y_pred_label = torch.argmax(y_pred, dim=1)  # Get the predicted class labels

        # Store true and predicted labels for accuracy calculation
        train_labels.extend(label.cpu().detach().numpy())
        train_preds.extend(y_pred_label.cpu().detach().numpy())
        
        # Calculate the loss
        loss = criterion(y_pred, label)
        
        optimizer.zero_grad()  # Zero the gradients
        loss.backward()  # Backpropagate the loss
        optimizer.step()  # Update the model parameters

        train_running_loss += loss.item()  # Accumulate the loss

    # Calculate average training loss for this epoch
    train_loss = train_running_loss / (idx + 1)

    # Validation loop
    model.eval()  # Set the model to evaluation mode
    val_labels = []
    val_preds = []
    val_running_loss = 0
    
    with torch.no_grad():  # No gradient calculation during validation
        for idx, img_label in enumerate(tqdm(val_dataloader, position=0, leave=True)):
            img = img_label["image"].float().to(device)
            label = img_label["label"].type(torch.long).to(device)         
            y_pred = model(img)  # Get predictions for validation data
            y_pred_label = torch.argmax(y_pred, dim=1)  # Get predicted class labels
            
            # Store true and predicted labels for accuracy calculation
            val_labels.extend(label.cpu().detach().numpy())
            val_preds.extend(y_pred_label.cpu().detach().numpy())
            
            # Calculate the loss
            loss = criterion(y_pred, label)
            val_running_loss += loss.item()  # Accumulate the validation loss

    # Calculate average validation loss for this epoch
    val_loss = val_running_loss / (idx + 1)

    # Print training and validation statistics
    print("-"*30)
    print(f"Train Loss EPOCH {epoch+1}: {train_loss:.4f}")
    print(f"Valid Loss EPOCH {epoch+1}: {val_loss:.4f}")
    print(f"Train Accuracy EPOCH {epoch+1}: {sum(1 for x,y in zip(train_preds, train_labels) if x == y) / len(train_labels):.4f}")
    print(f"Valid Accuracy EPOCH {epoch+1}: {sum(1 for x,y in zip(val_preds, val_labels) if x == y) / len(val_labels):.4f}")
    print("-"*30)

# End timing the training process
stop = timeit.default_timer()
print(f"Training Time: {stop-start:.2f}s")

# Save the trained model
torch.save(model.state_dict(), "ViT_trained_model.pth")
