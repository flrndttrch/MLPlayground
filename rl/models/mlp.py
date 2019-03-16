import torch.nn.functional as F
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers=None, activation=F.relu):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation

        if not hidden_layers:
            hidden_layers = [32, 32]

        self.linear = nn.ModuleList()
        self.prev_layer = self.input_dim
        for layer in hidden_layers:
            self.linear.append(nn.Linear(self.prev_layer, layer))
            self.prev_layer = layer

        self.output = nn.Linear(self.prev_layer, self.output_dim)

    def forward(self, x):
        for l in self.linear:
            x = self.activation(l(x))

        return self.output(x)
