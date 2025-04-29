import torch
from torch import nn

import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

train_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor(),
    target_transform=None
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor(),
    target_transform=None
)

# print(len(train_data))
# print(len(test_data))

# # plot images
# class_names = train_data.classes
# torch.manual_seed(11)
# figure = plt.figure(figsize=(9,9))
# rows, cols = 5, 5
# for i in range(1, rows*cols + 1):
#     random_index = torch.randint(0, len(train_data), size=[1]).item()
#     img, label = train_data[random_index]
#     figure.add_subplot(rows, cols, i)
#     plt.imshow(img.squeeze(), cmap="gray")
#     plt.title(class_names[label])
#     plt.axis(False)

# print(random_index)

# Setup batch size hyper parameter
BATCH_SIZE = 32

# Turn Datasets to Iterables (Batches)
train_dataloader = DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_dataloader = DataLoader(
    dataset=test_data,
    batch_size=BATCH_SIZE,
    shuffle=False
)

train_features_batch, train_labels_batch = next(iter(train_dataloader))