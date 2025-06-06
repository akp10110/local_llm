import torch
import FashionMNISTModelV0
from torch import nn
from timeit import default_timer as timer
from tqdm.auto import tqdm
import data_sets as data
import sys
import os
# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import helpers
import random

model_0 = FashionMNISTModelV0.FashionMNISTModelV0(
    input_features=784, # this is 28 * 28 - the lenght and width in pixels for the images
    output_features=10, # One for each class, so total 10
    hidden_units=10) # Randomly put as 10

# Create loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_0.parameters(), 
                            lr=0.1)

# Accuracy function - what % does our model get right?
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    accuracy = (correct/len(y_pred))*100
    return accuracy

def print_train_time(start: float, end: float):
    total_time =  end - start
    print(f"Train time - {total_time:.3f} seconds")
    return total_time

# Set the seed and start the timer
torch.manual_seed(42)
train_time_start = timer()

# Set the number of epochs
epochs = 3

# Loop through data
for epoch in tqdm(range(epochs)):

    ## Training
    train_loss = 0
    
    # Loop through training batches
    for batch, (X, y) in enumerate(data.train_dataloader):
        model_0.train()

        # Forward pass
        y_pred = model_0(X)

        # Calculate the loss
        loss = loss_fn(y_pred, y)
        train_loss += loss

        # Optimizer zero grad
        optimizer.zero_grad()

        # Back propagation
        loss.backward()

        # Optimizer step
        optimizer.step()

        # Print out whats happening
        if batch % 200 == 0:
            print(f"Looked at {batch * len(X)}/{len(data.train_dataloader.dataset)} samples")

    # Divide total train loss by length of train loader
    train_loss /= len(data.train_dataloader)

    ## Testing
    test_loss, test_accuracy = 0, 0
    model_0.eval()

    with torch.inference_mode():
        for X_text, y_test in data.test_dataloader:

            # Forward pass
            test_pred = model_0(X_text)
            test_pred_labels = test_pred.argmax(dim=1)

            # Calculate the loss/accuracy
            test_loss += loss_fn(test_pred, y_test)
            test_accuracy += accuracy_fn(y_true=y_test, 
                                    y_pred=test_pred_labels)
            
        # Calculate the test loss average per batch
        test_loss /= len(data.test_dataloader)

        # Calculate the test accuracy average per batch
        test_accuracy /= len(data.test_dataloader)

    # Print out what's happening
    print(f"Train loss : {train_loss:.4f} | Test loss : {test_loss:.4f} | Test Accuracy : {test_accuracy:.4f}")

train_time_end = timer()
total_train_time_model_0 = print_train_time(start=train_time_start, end=train_time_end)

model_0_results = helpers.eval_model(model=model_0,
                                     data_loader=data.test_dataloader,
                                     loss_fn=loss_fn,
                                     accuracy_fn=accuracy_fn)

print(model_0_results)
#### Results ####
# Train time - 22.295 seconds
# {'model_name': 'FashionMNISTModelV0', 'model_loss': 0.479365736246109, 'model_acc': 83.33666134185303}

# Plot predictions
for i in range (0,4): 
    test_samples = []
    test_labels = []
    for sample, label in random.sample(list(data.test_data), k=16):
        test_samples.append(sample)
        test_labels.append(label)

    pred_probs = helpers.make_predictions(model=model_0, data=test_samples)
    helpers.plot_predictions(pred_probs=pred_probs, 
                            test_samples=test_samples, 
                            test_labels=test_labels, 
                            class_names=data.class_names,
                            plt_main_title=f"Training Time - {total_train_time_model_0:.2f} secs || Accuracy - {model_0_results['model_acc']:.2f}%")