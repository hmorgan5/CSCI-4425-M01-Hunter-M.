# GenAI Acknowledgment
# Used Microsoft Copilot (M365 Copilot, GPT-5 based model)
# Purpose:
# - Debugged PyTorch tensor conversion issues.
# - Received guidance on implementing data preprocessing,
#   model training, evaluation metrics, README generation,
#   and loss curve plotting.
# - All code was reviewed, tested, and integrated by the student.

#### Part 1 ####
# Import necessary libraries
import pandas as pd
from sklearn.datasets import fetch_california_housing

##### Part 2 ####
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#### Part 3 ####
from sklearn.linear_model import LinearRegression
import torch
import torch.nn as nn
import torch.optim as optim

# ADDED because I get a seg fault otherwise
torch.set_num_threads(1)

#### Part 4 ####
from sklearn.metrics import mean_squared_error

#### Part 1 ####
# Load the California housing dataset
housingData = fetch_california_housing(as_frame=True)

# Print keys of the dataset
print(housingData.keys())

# Print the description of the dataset
print(housingData.DESCR)

# Creates a DataFrame from the data
dataframe = pd.DataFrame(housingData.data, columns=housingData.feature_names)

# Add the target variable to the DataFrame
dataframe["MedHouseVal"] = housingData.target

# Print the first 5 rows of the DataFrame
print(dataframe.head())

# Print summary statistics of the DataFrame
print(dataframe.describe())

# Saves the dataset description, first 5 rows, and summary statistics to a README.md file
with open("README.md", "w") as f:

    f.write("# California Housing Dataset Exploration\n\n")

    f.write("## How to Run\n\n")

    f.write("## Required Packages\n\n")
    f.write("- pandas\n")
    f.write("- scikit-learn\n")
    f.write("- torch\n")
    f.write("- matplotlib\n")
    f.write("- tabulate\n\n")

    f.write("1. Install the required libraries:\n\n")

    f.write("```bash\n")
    f.write("pip install pandas scikit-learn torch matplotlib tabulate\n")
    f.write("```\n\n")

    f.write("2. Run the Python script:\n\n")

    f.write("```bash\n")
    f.write("python hw2.py\n")
    f.write("```\n\n")

    f.write("3. The program will:\n")
    f.write("   - Load and explore the California Housing dataset.\n")
    f.write("   - Split and scale the data.\n")
    f.write("   - Train a Linear Regression model.\n")
    f.write("   - Train a PyTorch Neural Network model.\n")
    f.write("   - Evaluate both models using MSE and RMSE.\n")
    f.write("   - Generate a loss plot (lossPlot.png).\n")
    f.write("   - Generate this README.md file.\n\n")

    f.write("---\n\n")

    f.write("## Dataset Description\n\n")
    f.write("```\n")
    f.write(housingData.DESCR)
    f.write("\n```\n\n")

    f.write("## First 5 Rows\n\n")
    #f.write(dataframe.head().to_markdown())
    f.write("```\n")
    f.write(dataframe.head().to_string())
    f.write("\n```\n\n")
    f.write("\n\n")

    f.write("## Summary Statistics\n\n")
    #f.write(dataframe.describe().to_markdown())
    f.write("```\n")
    f.write(dataframe.describe().to_string())
    f.write("\n```\n\n")

#### Part 2 ####

# Seperates the features and target variable
X = dataframe.drop("MedHouseVal", axis=1)
y = dataframe["MedHouseVal"]

# Splits the dataset into training and testing sets with 80/20 split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Standardizes the features using StandardScaler
scaler = StandardScaler()

# Fits the scaler on the training data 
scaler.fit(X_train)

# Transforms the training and testing data using the fitted scaler
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# print the size of the training and testing sets
print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)

print("\nScaled Training Data:")
print(X_train_scaled[:5])

#### Part 3 ####
# Model 1
# Train a Linear Regression model
linearModel = LinearRegression()

# Fit the model on the scaled training data
linearModel.fit(X_train_scaled, y_train)

# Print message that model has been trained
print("\nLinear Regression Model trained.")

# Import numpy
import numpy as np

# Transform the scaled training and testing data into PyTorch tensors
X_train_scaled = np.array(X_train_scaled, dtype=np.float32, copy=True)
X_test_scaled = np.array(X_test_scaled, dtype=np.float32, copy=True)

# Convert the scaled training and testing data into PyTorch tensors
X_train_tensor = torch.from_numpy(X_train_scaled)
X_test_tensor = torch.from_numpy(X_test_scaled)

# Convert the target variable into PyTorch tensors
y_train_tensor = torch.from_numpy(y_train.to_numpy().astype(np.float32)).reshape(-1, 1)
y_test_tensor = torch.from_numpy(y_test.to_numpy().astype(np.float32)).reshape(-1, 1)

# Define a simple feedforward neural network model
class MLP (nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

# Get the input size for the neural network
input_size = X_train.shape[1]

# Instantiate the neural network model
model = MLP(input_size)

# Print the model architecture
print("\nNeural Network Architecture:")
print(model)

# Define the loss function and optimizer
criterion = nn.MSELoss()

# Define the optimizer (Adam) with a learning rate of 0.01
optimizer = optim.Adam(model.parameters(),lr=0.01)

# Establish the number of epochs for training
epochs = 100

# Initialize a list to store the loss history for plotting
loss_history = []   

# Train the neural network model
for epoch in range(epochs):
    # Forward pass
    predictions = model(X_train_tensor)

    loss = criterion(predictions, y_train_tensor)

    loss_history.append(loss.item())

    # Clear old gradients 
    optimizer.zero_grad()

    # Backpropagate
    loss.backward()

    # Update weights
    optimizer.step()

    # Print every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], " f"Loss: {loss.item():.4f}")

#### Part 4 ####
# Linear Regression #
linear_predictions = linearModel.predict(X_test_scaled)

# Calculate Mean Squared Error (MSE) and Root Mean Squared Error (RMSE) for Linear Regression
linear_mse = mean_squared_error(y_test, linear_predictions)

# Calculate RMSE for Linear Regression
linear_rmse = linear_mse ** 0.5

# Print the evaluation results for Linear Regression
print("\nLinear Regression Results")
print("MSE:", linear_mse)
print("RMSE:", linear_rmse)

# Neural Network #
# Set the model to evaluation mode and disable gradient calculation for inference
model.eval()
with torch.no_grad():
    nn_predictions = model(X_test_tensor)

# Convert the predictions from PyTorch tensor to NumPy array for evaluation
nn_predictions = nn_predictions.numpy().flatten()


# Calculate Mean Squared Error (MSE) and Root Mean Squared Error (RMSE) for Neural Network
nn_mse = mean_squared_error(y_test, nn_predictions)

# Calculate RMSE for Neural Network
nn_rmse = nn_mse ** 0.5

# Print the evaluation results for Neural Network
print("\nNeural Network Results")
print("MSE:", nn_mse)
print("RMSE:", nn_rmse)

# Save the evaluation results to the README.md file
with open("README.md", "a") as f:

    f.write("\n\n# Model Evaluation\n\n")

    f.write("| Model | MSE | RMSE |\n")
    f.write("|-------|------|------|\n")
    f.write(f"| Linear Regression | {linear_mse:.4f} | {linear_rmse:.4f} |\n")
    f.write(f"| Neural Network | {nn_mse:.4f} | {nn_rmse:.4f} |\n\n")

    f.write("## Analysis\n\n")

    if nn_rmse < linear_rmse:
        f.write(
            "The Neural Network performed better because it achieved "
            "lower MSE and RMSE values on the test set. Lower values "
            "indicate that the model's predictions were closer to the "
            "actual house values.\n\n"
        )
    else:
        f.write(
            "The Linear Regression model performed better because it "
            "achieved lower MSE and RMSE values on the test set. Lower "
            "values indicate more accurate predictions.\n\n"
        )

#### Part 5 ####
import matplotlib.pyplot as plt

# Plot the training loss curve for the neural network
plt.figure(figsize=(8, 5))
plt.plot(loss_history)
plt.title("Neural Network Training Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True)

# Save the loss curve plot as a PNG file
plt.tight_layout()
plt.savefig("lossPlot.png")
plt.close()


with open("README.md", "a") as f:

    f.write("## Neural Network Loss Curve\n\n")
    f.write("![Neural Network Loss Curve](lossPlot.png)\n\n")

    f.write("### Loss Analysis\n\n")
    f.write(
        "The training loss decreased as the number of epochs increased, "
        "indicating that the neural network was learning from the training "
        "data. The largest reduction in loss occurred during the early "
        "epochs, followed by smaller improvements as training continued. "
        "This behavior suggests that the model was converging toward a "
        "stable solution and improving its predictive performance over time.\n"
    )