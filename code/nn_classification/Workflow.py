import CircleModelV0
import torch
from torch import nn
import data_classification as data

model_0 = CircleModelV0.CircleModelV0()
print(f"=== model_0.state_dict() === \n {model_0.state_dict()}\n\n")


# Setup Loss function and optimizer
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1)

# Accuracy function - what % does our model get right?
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    accuracy = (correct/len(y_pred))*100
    return accuracy

# # View the first 5 outputs of the forward pass on the test data
# model_0.eval()
# with torch.inference_mode(): 
#     y_logits = model_0(data.X_test)[:5]
# print(f"=== First 5 logits === \n {y_logits} \n\n")

# # Use the sigmoid activation function on the logits to turn them into prediction probabilities
# y_pred_probs = torch.sigmoid(y_logits)
# print(f"=== y_pred_probs === \n {y_pred_probs} \n\n")

# y_pred_labels = torch.round(y_pred_probs)
# print(f"=== y_pred_labels === \n {y_pred_labels} \n\n")

# print(f"=== First 5 y_test values === \n {data.y_test[:5]} \n\n")

# ===== Training and Evaluation =====
torch.manual_seed(42)

# Set the number of epochs
epochs = 100

# Build training and evaluation loop
for epoch in range(epochs):
    ## Training
    model_0.train()

    # 1. Forward Pass
    y_logits = model_0(data.X_train).squeeze()
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
    model_0.eval()
    with torch.inference_mode():
        
        # 1. Forward Pass
        test_logits = model_0(data.X_test).squeeze()
        test_pred_labels = torch.round(torch.sigmoid(test_logits))

        # 2. Calculate test loss/accuracy
        test_loss = loss_fn(test_logits, test_pred_labels)
        test_accuracy = accuracy_fn(y_true=data.y_test, y_pred=test_pred_labels)

        # Print out whats happening
        if epoch % 5 == 0:
            print(f"Epoch: {epoch} | Loss: {loss:.5f} | Accuracy: {accuracy:.2f}% | Test Loss: {test_loss:.5f} | Test Accuracy: {test_accuracy:.2f}%")