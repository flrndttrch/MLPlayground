import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from rl import use_gpu
from rl.models.dqn import DQN
from rl.policies.eps_greedy import EpsGreedy


class DqnAgent():
    def __init__(self, state_dim, action_dim, lr=1e-3, l2_reg=1e-3, hidden_layers=None, activation=F.relu, gamma=1.0,
                 **eps_params):
        self.state_dim = state_dim
        self.action_dim = action_dim

        if hidden_layers is None:
            hidden_layers = [32, 32]
        self.gamma = gamma

        self.dqn = DQN(self.state_dim, self.action_dim, hidden_layers, activation=activation)
        self.dqn_target = DQN(self.state_dim, self.action_dim, hidden_layers, activation=activation)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        self.epsilon_greedy = EpsGreedy(**eps_params)

        self.optimizer = Adam(self.dqn.parameters(), lr=lr, weight_decay=l2_reg)

    def act(self, obs, t):
        if random.random() < self.epsilon_greedy.value(t):
            return random.randint(0, self.action_dim - 1)

        obs_t = torch.from_numpy(obs).float()
        if use_gpu():
            obs_t = obs_t.cuda()

        q = self.dqn.forward(obs_t)
        max_q, action = torch.max(q, 0)
        return action.data.cpu().numpy()

    def optimize(self, batch):
        states = torch.from_numpy(np.array(batch["states"], dtype=np.float32))
        actions = torch.from_numpy(np.array(batch["actions"], dtype=np.int64))
        rewards = torch.from_numpy(np.array(batch["rewards"], dtype=np.float32))
        next_states = torch.from_numpy(np.array(batch["next_states"], dtype=np.float32))
        masks = torch.from_numpy(np.array(batch["masks"], dtype=np.float32))

        if use_gpu():
            states = states.cuda()
            actions = actions.cuda()
            rewards = rewards.cuda()
            next_states = next_states.cuda()
            masks = masks.cuda()

        self.step(states, actions, rewards, next_states, masks)

    def step(self, states, actions, rewards, next_states, masks):
        state_action_values = self.dqn(states).gather(1, actions.unsqueeze(1))

        next_state_values = self.dqn_target(next_states).max(1)[0].detach()

        pred_state_action_values = (next_state_values * masks * self.gamma) + rewards

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, pred_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.dqn.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def update_target(self):
        self.dqn_target.load_state_dict(self.dqn.state_dict())