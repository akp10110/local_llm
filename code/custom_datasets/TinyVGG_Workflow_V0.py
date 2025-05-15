import random
import TinyVGGModelV0
import torch
from torch import nn
from timeit import default_timer as timer
from tqdm.auto import tqdm
import custom_data_sets as data
import sys
import os
# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import helpers
from datetime import datetime

model_0 = TinyVGGModelV0.TinyVGGModelV0(
    input_features=3, # color channels
    output_features=3, # number of classes
    hidden_units=10
)

# Count total trainable parameters
total_params = sum(p.numel() for p in model_0.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params}")

# Create loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_0.parameters(), 
                            lr=0.001)
# optimizer = torch.optim.SGD(params=model_0.parameters(), 
#                             lr=0.1)

# Set the seed and start the timer
torch.manual_seed(11)
train_time_start = timer()

# Set the number of epochs
epochs = 10

# Loop through data
for epoch in tqdm(range(epochs)):
    # Training
    helpers.model_training(model=model_0,
                           data_loader=data.train_data_loade_simple,
                           loss_fn=loss_fn,
                           optimizer=optimizer,
                           accuracy_fn=helpers.accuracy_fn)
    # Testing
    helpers.model_testing(model=model_0,
                          data_loader= data.test_data_loader_simple,
                          loss_fn=loss_fn,
                          optimizer=optimizer,
                          accuracy_fn=helpers.accuracy_fn)
    
train_time_end = timer()
total_train_time_model_0 = helpers.print_train_time(start=train_time_start, end=train_time_end)

model_0_results = helpers.eval_model(model=model_0,
                                     data_loader=data.test_data_loader_simple,
                                     loss_fn=loss_fn,
                                     accuracy_fn=helpers.accuracy_fn)

print(model_0_results)
#### Results ####

##### Save the model #####
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
model_path = f"trained_models/tinyvgg__epochs_10__opt_adam__pizza_steak_sushi_100percent_{timestamp}.pth"
torch.save(model_0.state_dict(), model_path)

# Plot predictions
for i in range (0,4): 
    test_samples = []
    test_labels = []
    for sample, label in random.sample(list(data.test_data_simple), k=16):
        test_samples.append(sample)
        test_labels.append(label)

    pred_probs = helpers.make_predictions(model=model_0, data=test_samples)
    helpers.plot_predictions(pred_probs=pred_probs, 
                            test_samples=test_samples, 
                            test_labels=test_labels, 
                            class_names=data.class_names,
                            plt_main_title=f"Training Time - {total_train_time_model_0:.2f} secs || Accuracy - {model_0_results['model_acc']:.2f}%")