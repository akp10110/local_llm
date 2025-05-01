import torch

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
