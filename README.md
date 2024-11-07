# ViT (Vision Transformer) Implementation With PyTorch

<div align="center">
    <a href="">
        <img alt="open-source-image"
        src="https://img.shields.io/badge/%E2%9D%A4%EF%B8%8F_Open_Source-%2350C878?style=for-the-badge"/>
    </a>
</div>
<br/>
<div align="center">
    <p>Liked my work? give this a ⭐!</p>
</div>

<p align="center">
  <img src="https://github.com/uygarkurt/ViT-PyTorch/blob/main/assets/arc.png" height="70%" width="70%"/>
</p>

This repository contains unofficial implementation of ViT (Vision Transformer) that is introduced in the paper [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) using PyTorch. Implementation has tested using the for image classification task using caltech-256 dataset.

## Table of Contents

- [ViT Implementation](#vitimp)
  - [ViT](#vit)
  - [PatchEmbedding](#embed)
- [Train Loop](#trainloop)
- [Inference](#inference)
- [Usage](#usage)
- [Contact](#contact)

## ViT Implementation <a class="anchor" id="vitimp"></a>

We need two classes to implement ViT. First is the `PatchEmbedding` to processing the image and embeddings until we feed the transformer encoder Second is the `ViT` for the rest of the process.

### ViT <a class="anchor" id="vit">

```
class ViT(nn.Module):
    def __init__(
        self,
        num_patches,
        num_classes,
        patch_size,
        embed_dim,
        num_heads,
        hidden_dim,
        num_layers,
        dropout,
        activation,
        in_channels,
    ):
        super().__init__()
        self.patch_embedding = PatchEmbedding(
            embed_dim, patch_size, num_patches, dropout, in_channels
        )
        self.encode = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                activation=activation,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=num_layers,
        )
        self.mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.encode(x)
        x = self.mlp(x[:, 0, :])
        return x
```

### PatchEmbedding <a class="anchor" id="embed">

```
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
```

## Train Loop <a class="anchor" id="trainloop"></a>

```
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), betas=ADAM_BETAS, lr=LEARNING_RATE, weight_decay=ADAM_WEIGHT_DECAY)

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
```

## Inference <a class="anchor" id="inference"></a>

```
plt.figure()
f, axarr = plt.subplots(2, 3)
counter = 0
for i in range(2):
    for j in range(3):
        axarr[i][j].imshow(imgs[counter].squeeze(), cmap="gray")
        axarr[i][j].set_title(f"Predicted {labels[counter]}")
        counter += 1
```

## Usage <a class="anchor" id="usage"></a>

You can run the code by downloading the file and updating the variables `train_df` and `test_df` to point a valid dataset location.

### This Readme file and the code is inspired by this repo by [uygarkurt](https://github.com/uygarkurt/ViT-PyTorch)
