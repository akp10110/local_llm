import torch
import FashionMNISTModelV1
from torch import nn
from timeit import default_timer as timer
from tqdm.auto import tqdm
import data_sets as data
import sys
import os
# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import helpers

model_1 = FashionMNISTModelV1.FashionMNISTModelV1(
    input_features=784,
    output_features=10,
    hidden_units=10
)

# Create loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_1.parameters(), 
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
    # Training
    helpers.model_training(model=model_1,
                           data_loader=data.train_dataloader,
                           loss_fn=loss_fn,
                           optimizer=optimizer,
                           accuracy_fn=accuracy_fn)
    # Testing
    helpers.model_testing(model=model_1,
                          data_loader= data.test_dataloader,
                          loss_fn=loss_fn,
                          optimizer=optimizer,
                          accuracy_fn=accuracy_fn)
    
train_time_end = timer()
total_train_time_model_1 = print_train_time(start=train_time_start, end=train_time_end)

model_1_results = helpers.eval_model(model=model_1,
                                     data_loader=data.test_dataloader,
                                     loss_fn=loss_fn,
                                     accuracy_fn=accuracy_fn)

print(model_1_results)
# {'model_name': 'FashionMNISTModelV1', 'model_loss': 0.515899658203125, 'model_acc': 82.10862619808307}
