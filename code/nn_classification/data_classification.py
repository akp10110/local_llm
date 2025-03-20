from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import torch

# Make 1000 samples
n_samples = 1000

# Create circles
X, y = make_circles(n_samples=n_samples, noise=0.01, random_state=42)
# print(f"First five of X = {X[:5]}")
# print(f"First five of y = {y[:5]}")

# Make dataframe of circle data
# circles = pd.DataFrame({"X1": X[:, 0], 
#                         "X2": X[:, 1], 
#                         "label": y})
# print(circles.head(20))

# Plot the circles
# plt.scatter(x=X[:, 0], 
#             y=X[:, 1], 
#             c=y, 
#             cmap=plt.cm.BrBG)
# print("Pause 1")

## Turn data into tensors and create train and test splits
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2,
                                                    random_state=42)
