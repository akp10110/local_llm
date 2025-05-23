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

# Set the seed and start the timer
torch.manual_seed(11)
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
                           accuracy_fn=helpers.accuracy_fn)
    # Testing
    helpers.model_testing(model=model_1,
                          data_loader= data.test_dataloader,
                          loss_fn=loss_fn,
                          optimizer=optimizer,
                          accuracy_fn=helpers.accuracy_fn)
    
train_time_end = timer()
total_train_time_model_1 = helpers.print_train_time(start=train_time_start, end=train_time_end)

model_1_results = helpers.eval_model(model=model_1,
                                     data_loader=data.test_dataloader,
                                     loss_fn=loss_fn,
                                     accuracy_fn=helpers.accuracy_fn)

print(model_1_results)
#### Results ####
# Train time - 22.544 seconds
# {'model_name': 'FashionMNISTModelV1', 'model_loss': 0.513516366481781, 'model_acc': 81.94888178913737}
