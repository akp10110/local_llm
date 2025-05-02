from torch import nn
import sys
import os
# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import helpers


class FashionMNISTModelV1(nn.Module):
    def __init__(self, input_features, output_features, hidden_units, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear_layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_features),
            nn.ReLU()
        )

    def forward(self, x):
        return self.linear_layer_stack(x)
