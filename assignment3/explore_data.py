# Generative AI acknowledgment:
# I used Microsoft 365 Copilot, based on the GPT-5 reasoning model,
# to help interpret the assignment requirements and create an initial
# structure for the data exploration code. I reviewed and tested the code.

import os

import matplotlib.pyplot as plt
import numpy as np

from data import get_raw_datasets, get_loaders


# Create the figures folder if it does not already exist.
os.makedirs("figures", exist_ok=True)


# ---------------------------------------------------------
# Part 1.2: Explore the raw dataset
# ---------------------------------------------------------

train_dataset, test_dataset = get_raw_datasets()

print("Number of training samples:", len(train_dataset))
print("Number of testing samples:", len(test_dataset))


# Retrieve the first sample.
image, (label, mask) = train_dataset[0]

print("\nFirst training sample:")
print("Image dimensions (width, height):", image.size)
print("Classification label:", label)
print("Mask dimensions (width, height):", mask.size)
print("Unique mask values:", np.unique(np.array(mask)))


# Convert the numeric classification label into readable text.
label_names = {
    0: "Cat",
    1: "Dog"
}

print("Classification name:", label_names[label])


# ---------------------------------------------------------
# Part 1.3: Visualize the first three samples
# ---------------------------------------------------------

fig, axes = plt.subplots(
    nrows=3,
    ncols=2,
    figsize=(8, 10)
)

for i in range(3):
    sample_image, (sample_label, sample_mask) = train_dataset[i]

    # Display the original image.
    axes[i, 0].imshow(sample_image)
    axes[i, 0].set_title(
        f"Image {i + 1}: {label_names[sample_label]}"
    )
    axes[i, 0].axis("off")

    # Display the corresponding segmentation mask.
    axes[i, 1].imshow(
        np.array(sample_mask),
        cmap="viridis",
        vmin=1,
        vmax=3
    )
    axes[i, 1].set_title(f"Segmentation Mask {i + 1}")
    axes[i, 1].axis("off")


plt.tight_layout()
plt.savefig(
    "figures/training_samples.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()


# ---------------------------------------------------------
# Part 1.4: Inspect one processed batch
# ---------------------------------------------------------

train_loader, test_loader = get_loaders(batch_size=16)

images, (labels, masks) = next(iter(train_loader))

print("\nProcessed training batch:")
print("Images shape:", images.shape)
print("Labels shape:", labels.shape)
print("Masks shape:", masks.shape)