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
torch.manual_seed(11)

# Create an instance of the model
model_0 = LinearRegressionModel.LinearRegressionModel()
print(model_0.state_dict())

with torch.inference_mode():
    y_predictions_set = model_0(X_test_set)
print(y_predictions_set)


plot_predictions(predictions=y_predictions_set);    
print("Stop")

