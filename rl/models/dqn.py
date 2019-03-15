import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_layers=None, duel=True, activation=F.relu):
        super(DQN, self).__init__()
        self.activation = activation
        self.input_dim = input_dim
        self.output_dim = output_dim

        if not hidden_layers:
            hidden_layers = [32, 32]

        self.linear = nn.ModuleList()
        prev_layer = self.input_dim
        for layer in hidden_layers:
            # TODO: insert duel architecture
            self.linear.append(nn.Linear(prev_layer, layer))
            prev_layer = layer
        self.output = nn.Linear(prev_layer, self.output_dim)

    def forward(self, x):
        for l in self.linear:
            x = self.activation(l(x))
        return self.output(x)
