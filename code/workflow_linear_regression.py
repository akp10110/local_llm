import torch
from torch import nn
import matplotlib.pyplot as plt

from classes import LinearRegressionModel

# Create known parameters
weight = 0.5
bias = 0.8

# Create data
start = 0
end = 1
step = 0.01
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

print(f"X = ", X)
print (f"y = ", y)

# Create train/test split
print("---- Create train/test split ----")
split = int (0.8 * len(X))
X_training_set = X[:split]
y_training_set = y[:split]

X_test_set = X[split:]
y_test_set = y[split:]

def plot_predictions(train_data = X_training_set, 
                     train_labels = y_training_set, 
                     test_data = X_test_set, 
                     test_labels = y_test_set, 
                     predictions = None):
    
    plt.figure(figsize=(8,6))

    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
  
    # Plot test data in green
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    if predictions is not None:
        # Plot the predictions in red (predictions were made on the test data)
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    # Show the legend
    plt.legend(prop={"size": 12})

# Create a random seed
torch.manual_seed(21)

# Create an instance of the model
model_0 = LinearRegressionModel.LinearRegressionModel()
print(model_0.state_dict())

# Original Prediction
with torch.inference_mode():
    y_predictions_set = model_0(X_test_set)
plot_predictions(predictions=y_predictions_set);    
print("Stop")

# Setup Loss function and Optimizer
loss_fn = torch.nn.L1Loss()
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)

### Training loop
epochs = 165
print(f"Original Parameters = {weight, bias}")

# Step 0: Loop through the data
for epoch in range(epochs):
    # Set the model to training mode
    model_0.train()

    # Step 1. Forward pass
    y_predictions_set = model_0(X_training_set)

    # Step 2: Calculate the loss
    loss = loss_fn(y_predictions_set, y_training_set)

    # Step 3: Optimiser zero grad
    optimizer.zero_grad()

    # Step 4: Perform back propagation
    loss.backward()

    # Step 5: Step the optimiser - perform gradient descent
    optimizer.step()

    print(f"Loss = {loss}")
    print(f"New Parameters = {model_0.state_dict()}")

    ### Testing
    model_0.eval()
    with torch.inference_mode():
        # Step 1: Do forward pass
        test_prediction_set = model_0(X_test_set)

        # Step 2: Calculate the loss
        test_loss = loss_fn(test_prediction_set, y_test_set)

    # Print what's happening
    print(f"Epoch: {epoch}, Loss: {loss}, Test Loss: {test_loss}")
    print(f"New Parameters = {model_0.state_dict()}")

# New Prediction
with torch.inference_mode():
    y_predictions_set_new = model_0(X_test_set)

plot_predictions(predictions=y_predictions_set_new);    
print("Stop 1")