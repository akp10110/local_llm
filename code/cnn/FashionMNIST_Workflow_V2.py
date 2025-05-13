import random
import FashionMNISTModelV2
import torch
from torch import nn
from timeit import default_timer as timer
from tqdm.auto import tqdm
import data_sets as data
import sys
import os
# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import helpers

model_2 = FashionMNISTModelV2.FashionMNISTModelV2(
    input_features=1,
    output_features=10,
    hidden_units=10
)

# Count total trainable parameters
total_params = sum(p.numel() for p in model_2.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params}")

# Create loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_2.parameters(), 
                            lr=0.1)

# Set the seed and start the timer
torch.manual_seed(11)
train_time_start = timer()

# Set the number of epochs
epochs = 3

# Loop through data
for epoch in tqdm(range(epochs)):
    # Training
    helpers.model_training(model=model_2,
                           data_loader=data.train_dataloader,
                           loss_fn=loss_fn,
                           optimizer=optimizer,
                           accuracy_fn=helpers.accuracy_fn)
    # Testing
    helpers.model_testing(model=model_2,
                          data_loader= data.test_dataloader,
                          loss_fn=loss_fn,
                          optimizer=optimizer,
                          accuracy_fn=helpers.accuracy_fn)
    
train_time_end = timer()
total_train_time_model_2 = helpers.print_train_time(start=train_time_start, end=train_time_end)

model_2_results = helpers.eval_model(model=model_2,
                                     data_loader=data.test_dataloader,
                                     loss_fn=loss_fn,
                                     accuracy_fn=helpers.accuracy_fn)

print(model_2_results)
#### Results ####
# Train time - 134.766 seconds
# {'model_name': 'FashionMNISTModelV2', 'model_loss': 0.335775226354599, 'model_acc': 87.8694089456869}

# Plot predictions
for i in range (0,4): 
    test_samples = []
    test_labels = []
    for sample, label in random.sample(list(data.test_data), k=16):
        test_samples.append(sample)
        test_labels.append(label)

    pred_probs = helpers.make_predictions(model=model_2, data=test_samples)
    helpers.plot_predictions(pred_probs=pred_probs, 
                            test_samples=test_samples, 
                            test_labels=test_labels, 
                            class_names=data.class_names,
                            plt_main_title=f"Training Time - {total_train_time_model_2:.2f} secs || Accuracy - {model_2_results['model_acc']:.2f}%")