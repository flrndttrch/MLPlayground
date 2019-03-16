import torch.nn as nn
import torch.nn.functional as F

from rl.models.mlp import MLP


class DQN(MLP):

    def __init__(self, state_dim, action_dim, hidden_layers=None, duel=True, activation=F.relu):
        super(DQN, self).__init__(input_dim=state_dim, output_dim=action_dim, hidden_layers=hidden_layers,
                                  activation=activation)
        self.activation = activation
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.duel = duel

        if self.duel:
            self.adv_layer1 = nn.Linear(self.prev_layer, self.prev_layer)
            self.adv_layer2 = nn.Linear(self.prev_layer, self.action_dim)
            self.val_layer1 = nn.Linear(self.prev_layer, self.prev_layer)
            self.val_layer2 = nn.Linear(self.prev_layer, 1)

    def forward(self, x):
        for l in self.linear:
            x = self.activation(l(x))

        if self.duel:
            adv = self.adv_layer2(self.activation(self.adv_layer1(x)))
            val = self.val_layer2(self.activation(self.val_layer1(x)))

            return val + adv - adv.mean()
        else:
            return self.output(x)
