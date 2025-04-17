import torch
import BlobModel
import data_multiclass_classification as Data
from torch import nn
import matplotlib.pyplot as plt
from helper_utils import plot_decision_boundary
import time

model_0 =  BlobModel.BlobModel(input_features=2, output_features=4)

# Create a loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_0.parameters(), 
                            lr=0.1)

# Accuracy function - what % does our model get right?
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    accuracy = (correct/len(y_pred))*100
    return accuracy

# Create a trianing and testing loop
torch.manual_seed(Data.RANDOM_SEED)

# Set number of epochs
epochs = 100

# Loop through data
for epoch in range(epochs):
    model_0.train()

    # 1. Forward Pass
    y_logits = model_0(Data.X_blob_train)
    y_pred_labels = torch.softmax(y_logits, dim=1).argmax(dim=1)

    # 2. Calculate the loss/accuracy
    loss = loss_fn(y_logits, Data.y_blob_train)
    accuracy = accuracy_fn(y_true=Data.y_blob_train,
                           y_pred=y_pred_labels)
    
    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backward
    loss.backward()

    # 5. Optimizer Step
    optimizer.step()


    ## Testing
    model_0.eval()
    with torch.inference_mode():

        # 1. Forward Pass
        test_logits = model_0(Data.X_blob_test)
        test_pred_labels = torch.softmax(test_logits, dim=1).argmax(dim=1)

        # 2. Calculate the loss/accuracy
        test_loss = loss_fn(test_logits, Data.y_blob_test)
        test_accuracy = accuracy_fn(y_true=Data.y_blob_test, 
                                    y_pred=test_pred_labels)
        
        # Print out whats happening
        if epoch % 5 == 0:
            print(f"Epoch: {epoch} | Loss: {loss:.5f} | Accuracy: {accuracy:.2f}% | Test Loss: {test_loss:.5f} | Test Accuracy: {test_accuracy:.2f}%")

        # # Plot decision boundary for training and test sets
        # if(epoch > 1 and epoch % 2 == 0):
        #     plt.figure(figsize=(12, 6))
        #     plt.subplot(1, 2, 1)
        #     train = f"Training data || Iteration : {epoch} || Accuracy : {accuracy:.2f}%"
        #     plt.title(train)
        #     plot_decision_boundary(model=model_0, X=Data.X_blob_train, y=Data.y_blob_train)
        #     plt.subplot(1, 2, 2)
        #     test = f"Testing data || Iteration : {epoch} || Accuracy : {test_accuracy:.2f}%"
        #     plt.title(test)
        #     plot_decision_boundary(model=model_0, X=Data.X_blob_test, y=Data.y_blob_test)
        #     print("1")

# Plot decision boundary for training and test sets
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model=model_0, X=Data.X_blob_train, y=Data.y_blob_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model=model_0, X=Data.X_blob_test, y=Data.y_blob_test)
print("1")