from torchvision import transforms
from torchvision import datasets
import os
from torch.utils.data import DataLoader
from pathlib import Path

# Images data path
data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi_100_percent"
train_dir = image_path / "train"
test_dir = image_path / "test"

# Batch Size and Number of Workers
BATCH_SIZE = 32
# NUM_WORKERS =  os.cpu_count()

# print(f"\nTrain Dir {train_dir} & Test Dir {test_dir}\n")
# print(f"OS CPU Count - {NUM_WORKERS}")

#Create simple transform
transform_simple = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# train_data_simple = datasets.ImageFolder(root=train_dir, transform=transform_simple)
# test_data_simple = datasets.ImageFolder(root=test_dir, transform=transform_simple)
train_data_simple = datasets.ImageFolder(root=train_dir, transform=train_transform)
test_data_simple = datasets.ImageFolder(root=test_dir, transform=test_transform)

class_names = train_data_simple.classes

# Create Data loaders
train_data_loade_simple = DataLoader(dataset=train_data_simple, 
                                     batch_size=BATCH_SIZE,
                                     shuffle=True)
                                    #  num_workers=NUM_WORKERS)
test_data_loader_simple = DataLoader(dataset=test_data_simple,
                                     batch_size=BATCH_SIZE,
                                     shuffle=False)
                                    #  num_workers=NUM_WORKERS)