# Generative AI acknowledgment:
# I used Microsoft 365 Copilot, based on the GPT-5 reasoning model,
# to help interpret the model requirements and structure the PyTorch
# model and training code. I reviewed and tested the code.

import torch
import torch.nn as nn
from torchvision.models import resnet18

import time
from sklearn.metrics import accuracy_score
from data import get_loaders


# ---------------------------------------------------------
# Reproducibility and device setup
# ---------------------------------------------------------

SEED = 0

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Random seed:", SEED)
print("CUDA available:", torch.cuda.is_available())
print("Selected device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ---------------------------------------------------------
# Training settings
# ---------------------------------------------------------

BATCH_SIZE = 16
LEARNING_RATE = 0.001
NUM_EPOCHS = 100


# ---------------------------------------------------------
# Regular CNN
# ---------------------------------------------------------

class RegularCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Linear(
            in_features=64,
            out_features=2
        )

    def forward(self, x):
        x = self.features(x)

        # Convert [B, 64, 1, 1] into [B, 64].
        x = torch.flatten(x, start_dim=1)

        # Produce two raw classification scores per image.
        x = self.classifier(x)

        return x


# ---------------------------------------------------------
# ResNet-18
# ---------------------------------------------------------

def create_resnet18():
    # weights=None means the model starts with random weights.
    model = resnet18(weights=None)

    number_of_features = model.fc.in_features

    # Replace the original output layer with a two-class layer.
    model.fc = nn.Linear(
        in_features=number_of_features,
        out_features=2
    )

    return model


# ---------------------------------------------------------
# Reusable U-Net convolution block
# ---------------------------------------------------------

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)


# ---------------------------------------------------------
# Small U-Net
# ---------------------------------------------------------

class SmallUNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder level 1
        self.encoder1 = DoubleConv(3, 16)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        # Encoder level 2
        self.encoder2 = DoubleConv(16, 32)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # Bottleneck
        self.bottleneck = DoubleConv(32, 64)

        # Decoder level 1
        self.upconv1 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=32,
            kernel_size=2,
            stride=2
        )

        # The input has 64 channels after concatenating:
        # 32 upsampled channels + 32 encoder channels.
        self.decoder1 = DoubleConv(64, 32)

        # Decoder level 2
        self.upconv2 = nn.ConvTranspose2d(
            in_channels=32,
            out_channels=16,
            kernel_size=2,
            stride=2
        )

        # The input has 32 channels after concatenating:
        # 16 upsampled channels + 16 encoder channels.
        self.decoder2 = DoubleConv(32, 16)

        # Produce three raw scores for each pixel:
        # 0 = pet, 1 = background, 2 = boundary.
        self.output_layer = nn.Conv2d(
            in_channels=16,
            out_channels=3,
            kernel_size=1
        )

    def forward(self, x):
        # Encoder level 1
        encoder1_output = self.encoder1(x)
        x = self.pool1(encoder1_output)

        # Encoder level 2
        encoder2_output = self.encoder2(x)
        x = self.pool2(encoder2_output)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder level 1
        x = self.upconv1(x)

        # Skip connection from encoder level 2.
        x = torch.cat(
            (x, encoder2_output),
            dim=1
        )

        x = self.decoder1(x)

        # Decoder level 2
        x = self.upconv2(x)

        # Skip connection from encoder level 1.
        x = torch.cat(
            (x, encoder1_output),
            dim=1
        )

        x = self.decoder2(x)

        # Return raw class scores without softmax.
        x = self.output_layer(x)

        return x

def train_classifier(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    number_of_epochs,
    model_name
):
    training_losses = []
    start_time = time.time()

    print(f"\nTraining {model_name}")

    for epoch in range(number_of_epochs):
        model.train()

        total_loss = 0.0
        total_images = 0

        for images, (labels, masks) in train_loader:
            # Classifiers use labels, not masks.
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            batch_size = images.size(0)

            total_loss += loss.item() * batch_size
            total_images += batch_size

        average_loss = total_loss / total_images
        training_losses.append(average_loss)

        print(
            f"{model_name} | "
            f"Epoch {epoch + 1}/{number_of_epochs} | "
            f"Average loss: {average_loss:.4f}"
        )

    elapsed_minutes = (time.time() - start_time) / 60

    print(
        f"{model_name} training time: "
        f"{elapsed_minutes:.2f} minutes"
    )

    return training_losses

def train_segmenter(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    number_of_epochs,
    model_name
):
    training_losses = []
    start_time = time.time()

    print(f"\nTraining {model_name}")

    for epoch in range(number_of_epochs):
        model.train()

        total_loss = 0.0
        total_images = 0

        for images, (labels, masks) in train_loader:
            # U-Net uses masks, not classification labels.
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, masks)

            loss.backward()
            optimizer.step()

            batch_size = images.size(0)

            total_loss += loss.item() * batch_size
            total_images += batch_size

        average_loss = total_loss / total_images
        training_losses.append(average_loss)

        print(
            f"{model_name} | "
            f"Epoch {epoch + 1}/{number_of_epochs} | "
            f"Average loss: {average_loss:.4f}"
        )

    elapsed_minutes = (time.time() - start_time) / 60

    print(
        f"{model_name} training time: "
        f"{elapsed_minutes:.2f} minutes"
    )

    return training_losses


def evaluate_classifier(model, test_loader, device, model_name):
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, (labels, masks) in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            all_predictions.extend(
                predictions.cpu().tolist()
            )

            all_labels.extend(
                labels.cpu().tolist()
            )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    print(f"{model_name} test accuracy: {accuracy:.4f}")

    return accuracy, all_predictions, all_labels

def calculate_majority_baseline(test_loader):
    all_labels = []

    for images, (labels, masks) in test_loader:
        all_labels.extend(labels.tolist())

    dog_count = sum(
        label == 1 for label in all_labels
    )

    baseline = dog_count / len(all_labels)

    print(f"Always-Dog baseline accuracy: {baseline:.4f}")

    return baseline

def evaluate_segmenter(model, test_loader, device):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, (labels, masks) in test_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            predicted_masks = outputs.argmax(dim=1)

            correct += (
                predicted_masks == masks
            ).sum().item()

            total += masks.numel()

    pixel_accuracy = correct / total

    print(
        f"U-Net pixel accuracy: "
        f"{pixel_accuracy:.4f}"
    )

    return pixel_accuracy




# ---------------------------------------------------------
# Test the model output shapes
# ---------------------------------------------------------

def main():
    train_loader, test_loader = get_loaders(
        batch_size=BATCH_SIZE
    )

    # -----------------------------------------------------
    # Create models
    # -----------------------------------------------------

    cnn_model = RegularCNN().to(device)
    resnet_model = create_resnet18().to(device)
    unet_model = SmallUNet().to(device)

    print("\nModels moved to:", device)

    # -----------------------------------------------------
    # Verify output shapes
    # -----------------------------------------------------

    images, (labels, masks) = next(iter(train_loader))
    images = images.to(device)

    with torch.no_grad():
        cnn_output = cnn_model(images)
        resnet_output = resnet_model(images)
        unet_output = unet_model(images)

    print("\nInput and target shapes:")
    print("Images:", images.shape)
    print("Labels:", labels.shape)
    print("Masks:", masks.shape)

    print("\nModel output shapes:")
    print("CNN:", cnn_output.shape)
    print("ResNet-18:", resnet_output.shape)
    print("U-Net:", unet_output.shape)

    # -----------------------------------------------------
    # Loss functions
    # -----------------------------------------------------

    cnn_criterion = nn.CrossEntropyLoss()
    resnet_criterion = nn.CrossEntropyLoss()
    unet_criterion = nn.CrossEntropyLoss()

    # -----------------------------------------------------
    # Separate optimizers
    # -----------------------------------------------------

    cnn_optimizer = torch.optim.Adam(
        cnn_model.parameters(),
        lr=LEARNING_RATE
    )

    resnet_optimizer = torch.optim.Adam(
        resnet_model.parameters(),
        lr=LEARNING_RATE
    )

    unet_optimizer = torch.optim.Adam(
        unet_model.parameters(),
        lr=LEARNING_RATE
    )

    # -----------------------------------------------------
    # Train regular CNN
    # -----------------------------------------------------

    cnn_losses = train_classifier(
        model=cnn_model,
        train_loader=train_loader,
        criterion=cnn_criterion,
        optimizer=cnn_optimizer,
        device=device,
        number_of_epochs=NUM_EPOCHS,
        model_name="Regular CNN"
    )

    torch.save(
        cnn_model.state_dict(),
        "cnn_model.pth"
    )

    # -----------------------------------------------------
    # Train ResNet-18
    # -----------------------------------------------------

    resnet_losses = train_classifier(
        model=resnet_model,
        train_loader=train_loader,
        criterion=resnet_criterion,
        optimizer=resnet_optimizer,
        device=device,
        number_of_epochs=NUM_EPOCHS,
        model_name="ResNet-18"
    )

    torch.save(
        resnet_model.state_dict(),
        "resnet18_model.pth"
    )

    # -----------------------------------------------------
    # Train U-Net
    # -----------------------------------------------------

    unet_losses = train_segmenter(
        model=unet_model,
        train_loader=train_loader,
        criterion=unet_criterion,
        optimizer=unet_optimizer,
        device=device,
        number_of_epochs=NUM_EPOCHS,
        model_name="Small U-Net"
    )

    torch.save(
        unet_model.state_dict(),
        "unet_model.pth"
    )

    # -----------------------------------------------------
    # Evaluate classifiers
    # -----------------------------------------------------

    cnn_accuracy, cnn_predictions, test_labels = (
        evaluate_classifier(
            model=cnn_model,
            test_loader=test_loader,
            device=device,
            model_name="Regular CNN"
        )
    )

    resnet_accuracy, resnet_predictions, _ = (
        evaluate_classifier(
            model=resnet_model,
            test_loader=test_loader,
            device=device,
            model_name="ResNet-18"
        )
    )

    majority_baseline = calculate_majority_baseline(
        test_loader
    )

    # -----------------------------------------------------
    # Evaluate segmentation
    # -----------------------------------------------------

    unet_pixel_accuracy = evaluate_segmenter(
        model=unet_model,
        test_loader=test_loader,
        device=device
    )

    # -----------------------------------------------------
    # Save training history and evaluation results
    # -----------------------------------------------------

    results = {
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "number_of_epochs": NUM_EPOCHS,
        "cnn_losses": cnn_losses,
        "resnet_losses": resnet_losses,
        "unet_losses": unet_losses,
        "cnn_accuracy": cnn_accuracy,
        "resnet_accuracy": resnet_accuracy,
        "majority_baseline": majority_baseline,
        "unet_pixel_accuracy": unet_pixel_accuracy,
        "cnn_predictions": cnn_predictions,
        "resnet_predictions": resnet_predictions,
        "test_labels": test_labels
    }

    torch.save(
        results,
        "training_results.pth"
    )

    print("\nFinal evaluation results:")
    print(
        f"Always-Dog baseline: "
        f"{majority_baseline:.4f}"
    )
    print(
        f"Regular CNN accuracy: "
        f"{cnn_accuracy:.4f}"
    )
    print(
        f"ResNet-18 accuracy: "
        f"{resnet_accuracy:.4f}"
    )
    print(
        f"U-Net pixel accuracy: "
        f"{unet_pixel_accuracy:.4f}"
    )

    print("\nModels and training results saved.")


if __name__ == "__main__":
    main()