from torch import nn

class FashionMNISTModelV0(nn.Module):
    def __init__(self, input_features, output_features, hidden_units, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear_layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_features),
            nn.ReLu()
        )

    def forward(self, x):
        return self.linear_layer_stack(x)
