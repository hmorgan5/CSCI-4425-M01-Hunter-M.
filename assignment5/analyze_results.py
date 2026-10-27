# Generative AI acknowledgment:
# I used Microsoft 365 Copilot, based on the GPT-5 reasoning model,
# to help structure the plotting code.
# I reviewed the code and generated the figures using my training results.

import json
from pathlib import Path

import matplotlib

# Allows figures to be generated on a remote server without a display.
matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ---------------------------------------------------------
# Setup paths
# ---------------------------------------------------------

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
RESULTS_FILE = SCRIPT_DIRECTORY / "training_results.json"
FIGURES_DIRECTORY = SCRIPT_DIRECTORY / "figures"

FIGURES_DIRECTORY.mkdir(exist_ok=True)


# ---------------------------------------------------------
# Load training results
# ---------------------------------------------------------

with RESULTS_FILE.open("r", encoding="utf-8") as file:
    results = json.load(file)

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

classification_figure = (
    FIGURES_DIRECTORY / "classification_training_loss.png"
)

plt.savefig(
    classification_figure,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Saved {classification_figure}")


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

unet_figure = FIGURES_DIRECTORY / "unet_training_loss.png"

plt.savefig(
    unet_figure,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Saved {unet_figure}")
print("Training-loss figures generated successfully.")