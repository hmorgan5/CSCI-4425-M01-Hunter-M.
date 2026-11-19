# GenAI acknowledgment:
# Microsoft Copilot powered by GPT-5.6 was used to generate, format,
# comment, and review this code.

import copy
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool


# Shared settings required or suggested by the assignment.
SEED = 0
EPOCHS = 100
BATCH_SIZE = 32
HIDDEN_CHANNELS = 64
DROPOUT = 0.5
LEARNING_RATE = 0.01

# Set random seeds so the experiment can be reproduced.
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Use a GPU if CUDA is available. Otherwise, train on the CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# =============================================================================
# Part 1: Data Loading and Exploration
# =============================================================================

# Load the PROTEINS graph-classification dataset with TUDataset.
dataset = TUDataset(
    root="data/TUDataset",
    name="PROTEINS",
)

# Inspect the first graph as required by the assignment.
first_graph = dataset[0]

# Calculate the average number of nodes across all graphs.
average_nodes = np.mean(
    [graph.num_nodes for graph in dataset]
)

# Report the number of graphs, classes, and node features.
print("\nDataset information")
print(f"Dataset: {dataset}")
print(f"Total number of graphs: {len(dataset)}")
print(f"Number of classes: {dataset.num_classes}")
print(f"Number of node features: {dataset.num_features}")

# Display the complete first graph object.
print("\nFirst graph")
print(first_graph)

# Display the first graph's node-feature matrix and dimensions.
print("\nNode feature matrix:")
print(first_graph.x)
print(f"x shape: {first_graph.x.shape}")

# Display the first graph's connectivity matrix and dimensions.
print("\nEdge index matrix:")
print(first_graph.edge_index)
print(f"edge_index shape: {first_graph.edge_index.shape}")

# Display the graph-level target label and average graph size.
print(f"\nGraph label: {first_graph.y}")
print(
    f"Average number of nodes per graph: "
    f"{average_nodes:.4f}"
)


# =============================================================================
# Part 2: Data Preprocessing
# =============================================================================

# Create one index and graph-level class label for every graph.
indices = np.arange(len(dataset))
labels = np.array(
    [
        int(dataset[index].y.item())
        for index in indices
    ]
)

# Reserve 20% of the complete dataset for final testing.
# random_state=0 provides the split required by the assignment.
train_validation_indices, test_indices = train_test_split(
    indices,
    test_size=0.20,
    random_state=SEED,
    shuffle=True,
    stratify=labels,
)

# Assign 12.5% of the remaining 80% to validation.
# This is approximately 10% of the complete dataset.
train_indices, validation_indices = train_test_split(
    train_validation_indices,
    test_size=0.125,
    random_state=SEED,
    shuffle=True,
    stratify=labels[train_validation_indices],
)

# Verify that no graph appears in more than one subset.
assert set(train_indices).isdisjoint(validation_indices)
assert set(train_indices).isdisjoint(test_indices)
assert set(validation_indices).isdisjoint(test_indices)

# Create the graph-level training, validation, and test datasets.
train_dataset = dataset[
    torch.tensor(train_indices, dtype=torch.long)
]
validation_dataset = dataset[
    torch.tensor(validation_indices, dtype=torch.long)
]
test_dataset = dataset[
    torch.tensor(test_indices, dtype=torch.long)
]

# Use a seeded generator for reproducible training-batch shuffling.
generator = torch.Generator().manual_seed(SEED)

# Shuffle only the training mini-batches.
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    generator=generator,
)

# Validation and test mini-batches do not need shuffling.
validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

# Report the actual graph counts and random seed.
print("\nDataset split")
print(f"Random seed: {SEED}")
print(f"Training graphs: {len(train_dataset)}")
print(f"Validation graphs: {len(validation_dataset)}")
print(f"Test graphs: {len(test_dataset)}")

# Inspect x, edge_index, batch, and y in one mini-batch.
sample_batch = next(iter(train_loader))

print("\nExample training mini-batch")
print(f"x shape: {sample_batch.x.shape}")
print(f"edge_index shape: {sample_batch.edge_index.shape}")
print(f"batch shape: {sample_batch.batch.shape}")
print(f"y shape: {sample_batch.y.shape}")


# =============================================================================
# Part 3: Model Building and Training
# =============================================================================

class GCNClassifier(nn.Module):
    """Two-layer GCN for graph-level classification."""

    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_classes,
    ):
        super().__init__()

        # Required first and second graph convolution layers.
        self.conv1 = GCNConv(
            in_channels,
            hidden_channels,
        )
        self.conv2 = GCNConv(
            hidden_channels,
            hidden_channels,
        )

        # Convert each pooled graph representation into class scores.
        self.output_layer = nn.Linear(
            hidden_channels,
            num_classes,
        )

    def forward(self, x, edge_index, batch):
        # First graph convolution, ReLU, and dropout.
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(
            x,
            p=DROPOUT,
            training=self.training,
        )

        # Second graph convolution and ReLU.
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Produce one representation for each graph in the batch.
        x = global_mean_pool(x, batch)

        # Return one score for each possible graph class.
        return self.output_layer(x)


class GATClassifier(nn.Module):
    """Two-layer GAT for graph-level classification."""

    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_classes,
        heads=4,
    ):
        super().__init__()

        # The first attention layer uses multiple attention heads.
        self.conv1 = GATConv(
            in_channels,
            hidden_channels,
            heads=heads,
        )

        # The first layer concatenates its heads, so the second
        # layer receives hidden_channels * heads input features.
        self.conv2 = GATConv(
            hidden_channels * heads,
            hidden_channels,
            heads=1,
            concat=False,
        )

        # Convert each pooled graph representation into class scores.
        self.output_layer = nn.Linear(
            hidden_channels,
            num_classes,
        )

    def forward(self, x, edge_index, batch):
        # First attention layer, ELU, and dropout.
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(
            x,
            p=DROPOUT,
            training=self.training,
        )

        # Second attention layer and ELU.
        x = self.conv2(x, edge_index)
        x = F.elu(x)

        # Produce one representation for each graph in the batch.
        x = global_mean_pool(x, batch)

        # Return one score for each possible graph class.
        return self.output_layer(x)


def calculate_validation_loss(
    model,
    data_loader,
    loss_function,
):
    """Calculate average validation loss without updating weights."""

    # Evaluation mode disables training-specific behavior such as dropout.
    model.eval()

    total_loss = 0.0
    total_graphs = 0

    # Validation must not calculate gradients or update parameters.
    with torch.no_grad():
        for batch_data in data_loader:
            batch_data = batch_data.to(device)

            output = model(
                batch_data.x,
                batch_data.edge_index,
                batch_data.batch,
            )

            loss = loss_function(
                output,
                batch_data.y.view(-1),
            )

            # Weight each batch loss by its number of graphs.
            total_loss += (
                loss.item() * batch_data.num_graphs
            )
            total_graphs += batch_data.num_graphs

    return total_loss / total_graphs


def train_model(model, model_name):
    """Train a model and retain its lowest-validation-loss checkpoint."""

    # Cross-entropy loss is appropriate for the two-class task.
    loss_function = nn.CrossEntropyLoss()

    # Use the Adam optimizer and learning rate specified in the setup.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # Record both losses for the required loss-curve figure.
    history = {
        "training_loss": [],
        "validation_loss": [],
    }

    best_validation_loss = float("inf")
    best_epoch = 0
    best_model_state = None

    for epoch in range(1, EPOCHS + 1):
        # Training mode enables dropout and parameter updates.
        model.train()

        total_training_loss = 0.0
        total_graphs = 0

        for batch_data in train_loader:
            batch_data = batch_data.to(device)

            # Clear gradients from the previous mini-batch.
            optimizer.zero_grad()

            # Perform the required forward pass.
            output = model(
                batch_data.x,
                batch_data.edge_index,
                batch_data.batch,
            )

            # Compare graph predictions with graph-level labels.
            loss = loss_function(
                output,
                batch_data.y.view(-1),
            )

            # Perform backpropagation and update the model parameters.
            loss.backward()
            optimizer.step()

            total_training_loss += (
                loss.item() * batch_data.num_graphs
            )
            total_graphs += batch_data.num_graphs

        # Calculate the average graph-weighted training loss.
        training_loss = (
            total_training_loss / total_graphs
        )

        # Use validation data for checkpoint selection only.
        validation_loss = calculate_validation_loss(
            model,
            validation_loader,
            loss_function,
        )

        history["training_loss"].append(training_loss)
        history["validation_loss"].append(validation_loss)

        # Retain the checkpoint with the lowest validation loss.
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_model_state = copy.deepcopy(
                model.state_dict()
            )

        # Print both losses every 10 epochs as required.
        if epoch % 10 == 0:
            print(
                f"{model_name} | "
                f"Epoch {epoch:03d} | "
                f"Training loss: {training_loss:.4f} | "
                f"Validation loss: {validation_loss:.4f}"
            )

    # Restore the validation-selected parameters before testing.
    model.load_state_dict(best_model_state)

    # Save the selected state and its validation information.
    checkpoint = {
        "model_state_dict": best_model_state,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
    }

    torch.save(
        checkpoint,
        f"{model_name.lower()}_best.pt",
    )

    print(
        f"\nBest {model_name} checkpoint: "
        f"epoch {best_epoch}, "
        f"validation loss {best_validation_loss:.4f}"
    )

    return (
        model,
        history,
        best_epoch,
        best_validation_loss,
    )


# Instantiate and train the GCN with the required dimensions.
gcn_model = GCNClassifier(
    in_channels=dataset.num_features,
    hidden_channels=HIDDEN_CHANNELS,
    num_classes=dataset.num_classes,
).to(device)

(
    gcn_model,
    gcn_history,
    gcn_best_epoch,
    gcn_best_loss,
) = train_model(gcn_model, "GCN")


# Instantiate and train the GAT on the same graph split.
gat_model = GATClassifier(
    in_channels=dataset.num_features,
    hidden_channels=HIDDEN_CHANNELS,
    num_classes=dataset.num_classes,
    heads=4,
).to(device)

(
    gat_model,
    gat_history,
    gat_best_epoch,
    gat_best_loss,
) = train_model(gat_model, "GAT")


# =============================================================================
# Part 4: Model Evaluation
# =============================================================================

def evaluate_model(model, data_loader):
    """Evaluate a selected checkpoint on the held-out test set."""

    model.eval()

    true_labels = []
    predicted_labels = []

    # Testing must not calculate gradients or update parameters.
    with torch.no_grad():
        for batch_data in data_loader:
            batch_data = batch_data.to(device)

            # Produce one set of class scores for each graph.
            output = model(
                batch_data.x,
                batch_data.edge_index,
                batch_data.batch,
            )

            # Select the class with the largest output score.
            predictions = output.argmax(dim=1)

            true_labels.extend(
                batch_data.y.view(-1).cpu().tolist()
            )
            predicted_labels.extend(
                predictions.cpu().tolist()
            )

    # Report all four metrics required by the assignment.
    return {
        "accuracy": accuracy_score(
            true_labels,
            predicted_labels,
        ),
        "precision": precision_score(
            true_labels,
            predicted_labels,
            zero_division=0,
        ),
        "recall": recall_score(
            true_labels,
            predicted_labels,
            zero_division=0,
        ),
        "f1_score": f1_score(
            true_labels,
            predicted_labels,
            zero_division=0,
        ),
    }


# Evaluate only after each model's checkpoint has been selected.
gcn_metrics = evaluate_model(
    gcn_model,
    test_loader,
)
gat_metrics = evaluate_model(
    gat_model,
    test_loader,
)

print("\nGCN test results")
print(f"Accuracy:  {gcn_metrics['accuracy']:.4f}")
print(f"Precision: {gcn_metrics['precision']:.4f}")
print(f"Recall:    {gcn_metrics['recall']:.4f}")
print(f"F1-score:  {gcn_metrics['f1_score']:.4f}")

print("\nGAT test results")
print(f"Accuracy:  {gat_metrics['accuracy']:.4f}")
print(f"Precision: {gat_metrics['precision']:.4f}")
print(f"Recall:    {gat_metrics['recall']:.4f}")
print(f"F1-score:  {gat_metrics['f1_score']:.4f}")


# =============================================================================
# Part 5: Analysis and Interpretation
# =============================================================================

# Plot the recorded training and validation losses for both models.
epochs = range(1, EPOCHS + 1)

plt.figure(figsize=(12, 7))

plt.plot(
    epochs,
    gcn_history["training_loss"],
    label="GCN training",
)

plt.plot(
    epochs,
    gcn_history["validation_loss"],
    label="GCN validation",
)

plt.plot(
    epochs,
    gat_history["training_loss"],
    label="GAT training",
)

plt.plot(
    epochs,
    gat_history["validation_loss"],
    label="GAT validation",
)

plt.xlabel("Epoch")
plt.ylabel("Cross-entropy loss")
plt.title("GCN and GAT Training and Validation Loss")
plt.legend()
plt.tight_layout()

# Save the figure so it can be displayed in README.md.
plt.savefig(
    "loss_curves.png",
    dpi=200,
)

plt.show()