from pathlib import Path
import random
from typing import List
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
import torchvision
from matplotlib import pyplot as plt
from torchvision import transforms


# Create transform pipleine to resize image
custom_image_transform = transforms.Compose([
    transforms.Resize((128, 128)),
])

def pred_and_plot_image(model: torch.nn.Module, 
                        image_path: str, 
                        class_names: List[str] = None, 
                        transform=None):
    """Makes a prediction on a target image and plots the image with its prediction."""
    
    # 1. Load in image and convert the tensor values to float32
    target_image = torchvision.io.read_image(str(image_path)).type(torch.float32)
    
    # 2. Divide the image pixel values by 255 to get them between [0, 1]
    target_image = target_image / 255. 
    
    # 3. Transform if necessary
    if transform:
        target_image = transform(target_image)
    
    # # 4. Make sure the model is on the target device
    # model.to(device)
    
    # 5. Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
        # Add an extra dimension to the image
        target_image = target_image.unsqueeze(dim=0)
    
        # Make a prediction on image with an extra dimension and send it to the target device
        target_image_pred = model(target_image)
        
    # 6. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # 7. Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)
    
    # 8. Plot the image alongside the prediction and prediction probability
    plt.imshow(target_image.squeeze().permute(1, 2, 0)) # make sure it's the right size for matplotlib
    if class_names:
        title = f"Pred: {class_names[target_image_pred_label.cpu()]} | Prob: {target_image_pred_probs.max().cpu()*100:.3f}%"
    else: 
        title = f"Pred: {target_image_pred_label} | Prob: {target_image_pred_probs.max().cpu()*100:.3f}%"
    plt.title(title)
    plt.axis(False)

# --------------------

saved_model_0 = TinyVGGModelV0.TinyVGGModelV0(
    input_features=3, # color channels
    output_features=3, # number of classes
    hidden_units=10
)

trained_models_path = Path("trained_models/")
model_path = trained_models_path / "tinyvgg__epochs_10__opt_adam__pizza_steak_sushi_100percent_2025-05-14_15-56-13.pth"
# "tinyvgg__epochs_10__opt_adam__pizza_steak_sushi_20percent_2025-05-14_14-40-34.pth"


saved_model_0.load_state_dict(torch.load(model_path))

data_path = Path("data/")
custom_image_path = data_path / "custom_images/pizza1.jpg"
class_names = data.class_names

# Pred on our custom image
pred_and_plot_image(model=saved_model_0,
                    image_path=custom_image_path,
                    class_names=class_names,
                    transform=custom_image_transform)
print("Test")

# Plot predictions
# for i in range (0,4): 
#     test_samples = []
#     test_labels = []
#     for sample, label in random.sample(list(data.test_data_simple), k=16):
#         test_samples.append(sample)
#         test_labels.append(label)

#     pred_probs = helpers.make_predictions(model=saved_model_0, data=test_samples)
#     helpers.plot_predictions(pred_probs=pred_probs, 
#                             test_samples=test_samples, 
#                             test_labels=test_labels, 
#                             class_names=data.class_names,
#                             plt_main_title=f"Title")
    
# print("Test1")