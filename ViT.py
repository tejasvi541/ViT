##
## Importing Libraries
##
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.data.util import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import timeit
from tqdm import tqdm


##
## Hyperparameters
##

RANDOM_SEED = 42
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
NUM_ENCODER = 6
EMBED_DIM = (PATCH_SIZE**2) * IN_CHANNELS  # 768
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2  # 196

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


##
## Embedding Layer
##


import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, embed_dim, patch_size, num_patches, dropout, in_channels):
        """
        Initializes a PatchEmbedding layer for the Vision Transformer.

        Parameters:
        - embed_dim: int, the dimension of the embedding space for each patch.
        - patch_size: int, the size of each patch (e.g., 16 means 16x16 patches).
        - num_patches: int, total number of patches in the image after splitting.
        - dropout: float, the dropout rate applied after positional embeddings.
        - in_channels: int, the number of input channels (e.g., 3 for RGB images).
        """
        super().__init__()

        # 1. Patch Embedding Layer: Splits the image into patches and projects each patch into embed_dim space.
        #    Using Conv2d with a kernel and stride equal to patch_size effectively splits the image.
        #    - in_channels: number of color channels in the image (e.g., 3 for RGB).
        #    - embed_dim: dimension of each patch embedding.
        #    - kernel_size and stride: patch_size, which controls the size of each patch.
        #    Output dimensions: [batch_size, embed_dim, num_patches^(1/2), num_patches^(1/2)]
        self.patcher = nn.Sequential(
            nn.Conv2d(
                in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
            ),
            nn.Flatten(
                2
            ),  # Flattens the last two dimensions into one, output: [batch_size, embed_dim, num_patches]
        )

        # 2. Class Token: A learnable parameter that serves as a summary of the entire input.
        #    It is prepended to the sequence of patch embeddings in the forward pass.
        #    Shape: [1, 1, embed_dim], expanded to [batch_size, 1, embed_dim] during forward pass.
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim), requires_grad=True)

        # 3. Positional Embedding: A learnable parameter that encodes positional information.
        #    - Shape: [1, num_patches + 1, embed_dim], where +1 accounts for the class token.
        #    - Provides each patch (including the class token) with a unique position in the sequence.
        self.positional_embedding = nn.Parameter(
            torch.randn(1, num_patches + 1, embed_dim), requires_grad=True
        )

        # 4. Dropout Layer: Applies dropout to the patch embeddings after positional encoding.
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass for the PatchEmbedding layer.

        Parameters:
        - x: Input image tensor with dimensions [batch_size, in_channels, height, width].

        Returns:
        - A tensor representing the input image as a sequence of patch embeddings with positional encoding.
        """

        # 1. Expand class token to match the batch size.
        #    - Shape after expansion: [batch_size, 1, embed_dim]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)

        # 2. Apply patch embedding convolution and flattening.
        #    - Conv2d splits image into patches and projects each to the embedding dimension.
        #    - Flatten reshapes to [batch_size, embed_dim, num_patches].
        #    - Permute changes dimensions to [batch_size, num_patches, embed_dim].
        x = self.patcher(x).permute(0, 2, 1)

        # 3. Concatenate the class token to the beginning of the sequence of patch embeddings.
        #    - Shape after concatenation: [batch_size, num_patches + 1, embed_dim]
        x = torch.cat([cls_token, x], dim=1)

        # 4. Add positional embeddings to each token in the sequence.
        #    - Positional embeddings give each patch a spatial reference.
        #    - Shape remains [batch_size, num_patches + 1, embed_dim] after addition.
        x = self.positional_embedding + x

        # 5. Apply dropout to the embeddings with positional encoding.
        x = self.dropout(x)

        return x
