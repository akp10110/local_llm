import torch
import matplotlib.pyplot as plt

# import sys
# import os
# # Add parent directory to path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

torch.manual_seed(11)

def eval_model(model: torch.nn.Module, 
               data_loader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               accuracy_fn):
    loss, accuracy = 0, 0
    model.eval()

    with torch.inference_mode():
        for X, y in data_loader:
            # Make predictions with the model
            y_pred = model(X)

            # Accumulate the loss and accuracy values per batch
            loss += loss_fn(y_pred, y)
            accuracy += accuracy_fn(y_true=y, 
                                y_pred=y_pred.argmax(dim=1))
       
        # Scale loss and acc to find the average loss/acc per batch
        loss /= len(data_loader)
        accuracy /= len(data_loader)
    
    return {"model_name": model.__class__.__name__, # only works when model was created with a class
            "model_loss": loss.item(),
            "model_acc": accuracy}

def model_training(model: torch.nn.Module,
                data_loader: torch.utils.data.DataLoader,
                loss_fn: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                accuracy_fn
                ):
     ## Training
    train_loss, train_accuracy = 0, 0
    model.train()

    # Loop through training batches
    for batch, (X, y) in enumerate(data_loader):
        # Forward pass
        y_pred = model(X)

        # Calculate the loss
        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_accuracy += accuracy_fn(y_true=y, 
                                 y_pred=y_pred.argmax(dim=1))
        
        # Optimizer zero grad
        optimizer.zero_grad()

        # Back propagation
        loss.backward()

        # Optimizer step
        optimizer.step()

    # Divide total train loss by length of train loader
    train_loss /= len(data_loader)
    train_accuracy /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_accuracy:.2f}%")

def model_testing(model: torch.nn.Module,
                data_loader: torch.utils.data.DataLoader,
                loss_fn: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                accuracy_fn
                ):
    test_loss, test_accuracy = 0, 0
    model.eval()

    with torch.inference_mode():
        for X, y in data_loader:

            # Forward pass
            test_pred = model(X)
            test_pred_labels = test_pred.argmax(dim=1)

            # Calculate the loss/accuracy
            test_loss += loss_fn(test_pred, y)
            test_accuracy += accuracy_fn(y_true=y, 
                                    y_pred=test_pred_labels)
            
        # Calculate the test loss average per batch
        test_loss /= len(data_loader)
        # Calculate the test accuracy average per batch
        test_accuracy /= len(data_loader)
        # Print out what's happening
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_accuracy:.2f}%\n")
    

# Accuracy function - what % does our model get right?
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    accuracy = (correct/len(y_pred))*100
    return accuracy


def print_train_time(start: float, end: float):
    total_time =  end - start
    print(f"Train time - {total_time:.3f} seconds")
    return total_time


def make_predictions(model: torch.nn.Module, data: list):
    pred_probs = []
    model.eval()
    with torch.inference_mode():
        for sample in data:
            sample = torch.unsqueeze(sample, dim=0)

            #Forward Pass
            logits = model(sample)

            # Get prediction probability
            pred_prob = torch.softmax(logits.squeeze(), dim=0)
            pred_probs.append(pred_prob)

    # Stack the pred_probs to turn list into a tensor
    return torch.stack(pred_probs)


def plot_predictions(
        pred_probs: torch.nn.Module,
        test_samples: list,
        test_labels: list, 
        class_names: list,
        plt_main_title: str):
    plt.figure(figsize=(13,9))
    rows = 4
    cols = 4

    pred_labels = pred_probs.argmax(dim=1)

    plt.suptitle(plt_main_title, fontsize=14)
    #plt.tight_layout(rect=[0, 0, 1, 0.95])

    for i, sample in enumerate(test_samples):
        # Create a subplot
        plt.subplot(rows, cols, i+1)

        # Plot the target image
        sample = sample.permute(1, 2, 0)
        plt.imshow(sample.squeeze(), cmap="gray")

        # Find the prediction label (in text form, e.g. "Sandal")
        pred_label = class_names[pred_labels[i]]

        # Get the truth label (in text form, e.g. "T-shirt")
        truth_label = class_names[test_labels[i]] 

        # Create the title text of the plot
        title_text = f"Pred: {pred_label} | Truth: {truth_label}"
        
        # Check for equality and change title colour accordingly
        if pred_label == truth_label:
            plt.title(title_text, fontsize=10, c="g") # green text if correct
        else:
            plt.title(title_text, fontsize=10, c="r") # red text if wrong
        plt.axis(False)


    print("test")
