# Generative AI acknowledgment:
# I used Microsoft 365 Copilot, based on the GPT-5 reasoning model,
# to help structure the plotting and segmentation-visualization code.
# I reviewed the code and generated the figures using my trained models.

import os

import matplotlib

# Allows figures to be generated on a remote server without a display.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch

from data import get_loaders
from train_models import SmallUNet


# ---------------------------------------------------------
# Setup
# ---------------------------------------------------------

BATCH_SIZE = 16

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Selected device:", device)

os.makedirs("figures", exist_ok=True)


# ---------------------------------------------------------
# Load training results
# ---------------------------------------------------------

results = torch.load(
    "training_results.pth",
    map_location="cpu",
    weights_only=False
)

cnn_losses = results["cnn_losses"]
resnet_losses = results["resnet_losses"]
unet_losses = results["unet_losses"]

print("CNN epochs:", len(cnn_losses))
print("ResNet-18 epochs:", len(resnet_losses))
print("U-Net epochs:", len(unet_losses))


# ---------------------------------------------------------
# Plot CNN and ResNet-18 training losses
# ---------------------------------------------------------

classification_epochs = range(1, len(cnn_losses) + 1)

plt.figure(figsize=(8, 5))

plt.plot(
    classification_epochs,
    cnn_losses,
    label="Regular CNN"
)

plt.plot(
    classification_epochs,
    resnet_losses,
    label="ResNet-18"
)

plt.xlabel("Epoch")
plt.ylabel("Average Training Loss")
plt.title("Classifier Training Loss")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "figures/classification_training_loss.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Saved figures/classification_training_loss.png"
)


# ---------------------------------------------------------
# Plot U-Net training loss separately
# ---------------------------------------------------------

unet_epochs = range(1, len(unet_losses) + 1)

plt.figure(figsize=(8, 5))

plt.plot(
    unet_epochs,
    unet_losses,
    label="Small U-Net"
)

plt.xlabel("Epoch")
plt.ylabel("Average Training Loss")
plt.title("U-Net Training Loss")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "figures/unet_training_loss.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved figures/unet_training_loss.png")


# ---------------------------------------------------------
# Load the trained U-Net
# ---------------------------------------------------------

unet_model = SmallUNet().to(device)

unet_state = torch.load(
    "unet_model.pth",
    map_location=device,
    weights_only=True
)

unet_model.load_state_dict(unet_state)
unet_model.eval()


# ---------------------------------------------------------
# Retrieve test images and predict their masks
# ---------------------------------------------------------

train_loader, test_loader = get_loaders(
    batch_size=BATCH_SIZE
)

images, (labels, masks) = next(iter(test_loader))

# Select the first three test samples.
sample_images = images[:3]
sample_masks = masks[:3]

with torch.no_grad():
    outputs = unet_model(sample_images.to(device))
    predicted_masks = outputs.argmax(dim=1)


# ---------------------------------------------------------
# Visualize segmentation results
# ---------------------------------------------------------

fig, axes = plt.subplots(
    nrows=3,
    ncols=3,
    figsize=(10, 10)
)

for index in range(3):
    image_array = (
        sample_images[index]
        .cpu()
        .permute(1, 2, 0)
        .numpy()
    )

    ground_truth_array = (
        sample_masks[index]
        .cpu()
        .numpy()
    )

    predicted_array = (
        predicted_masks[index]
        .cpu()
        .numpy()
    )

    # Original image
    axes[index, 0].imshow(image_array)
    axes[index, 0].set_title(
        f"Test Image {index + 1}"
    )
    axes[index, 0].axis("off")

    # Ground-truth mask
    axes[index, 1].imshow(
        ground_truth_array,
        cmap="viridis",
        vmin=0,
        vmax=2
    )
    axes[index, 1].set_title(
        f"Ground Truth {index + 1}"
    )
    axes[index, 1].axis("off")

    # Predicted mask
    axes[index, 2].imshow(
        predicted_array,
        cmap="viridis",
        vmin=0,
        vmax=2
    )
    axes[index, 2].set_title(
        f"Prediction {index + 1}"
    )
    axes[index, 2].axis("off")

plt.tight_layout()

plt.savefig(
    "figures/segmentation_results.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved figures/segmentation_results.png")
print("Analysis figures generated successfully.")