import CircleModelV2_NonLinear
import torch
from torch import nn
import data_classification as data
from helper_utils import plot_decision_boundary
import matplotlib.pyplot as plt

model_2 = CircleModelV2_NonLinear.CircleModelV2_NonLinear()

# Setup Loss function and optimizer
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=model_2.parameters(), lr=0.1)

# Accuracy function - what % does our model get right?
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    accuracy = (correct/len(y_pred))*100
    return accuracy

# ===== Training and Evaluation =====
torch.manual_seed(42)

# Set the number of epochs
epochs = 1000

# Build training and evaluation loop
for epoch in range(epochs):
    ## Training
    model_2.train()

    # 1. Forward Pass
    y_logits = model_2(data.X_train).squeeze()
    y_pred_labels = torch.round(torch.sigmoid(y_logits))

    # 2. Calculate the loss/accuracy
    loss = loss_fn(y_logits, data.y_train)
    accuracy = accuracy_fn(y_true=data.y_train, y_pred=y_pred_labels)

    # 3. Optimezer zero grad
    optimizer.zero_grad()

    # 4. Loss backward
    loss.backward()

    # 5. Optimizer step
    optimizer.step()

    ## Testing
    model_2.eval()
    with torch.inference_mode():
        
        # 1. Forward Pass
        test_logits = model_2(data.X_test).squeeze()
        test_pred_labels = torch.round(torch.sigmoid(test_logits))

        # 2. Calculate test loss/accuracy
        test_loss = loss_fn(test_logits, test_pred_labels)
        test_accuracy = accuracy_fn(y_true=data.y_test, y_pred=test_pred_labels)

        # Print out whats happening
        if epoch % 5 == 0:
            print(f"Epoch: {epoch} | Loss: {loss:.5f} | Accuracy: {accuracy:.2f}% | Test Loss: {test_loss:.5f} | Test Accuracy: {test_accuracy:.2f}%")

# Plot decision boundary for training and test sets
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model=model_2, X=data.X_train, y=data.y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model=model_2, X=data.X_test, y=data.y_test)
print("1")