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
    correct = torch.eq(y_true, y_pred).sum().item
    accuracy = (correct/len(y_pred))*100
    return accuracy

# View the first 5 outputs of the forward pass on the test data
model_0.eval()
with torch.inference_mode(): 
    y_logits = model_0(data.X_test)[:5]
print(f"=== First 5 logits === \n {y_logits} \n\n")

# Use the sigmoid activation function on the logits to turn them into prediction probabilities
y_pred_probs = torch.sigmoid(y_logits)
print(f"=== y_pred_probs === \n {y_pred_probs} \n\n")

y_pred_labels = torch.round(y_pred_probs)
print(f"=== y_pred_labels === \n {y_pred_labels} \n\n")

print(f"=== First 5 y_test values === \n {data.y_test[:5]} \n\n")