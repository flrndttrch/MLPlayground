import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from rl import use_gpu
from rl.memory.memory import PrioritizedReplayBuffer, ReplayBuffer
from rl.models.dqn import DQN
from rl.policies.eps_greedy import EpsGreedy


class DqnAgent():
    def __init__(self, state_dim, action_dim, lr=1e-4, l2_reg=1e-3, hidden_layers=None, activation=F.relu, gamma=1.0,
                 double=True, duel=True, loss_fct=F.mse_loss, mem_size=10000, mem_type='per', **eps_params):
        if hidden_layers is None:
            hidden_layers = [32, 32]

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.double = double
        self.loss_fct = loss_fct
        self.mem_type = mem_type

        if self.mem_type is 'per':
            self.memory = PrioritizedReplayBuffer(capacity=mem_size)
        else:
            self.memory = ReplayBuffer(capacity=mem_size)

        self.dqn = DQN(self.state_dim, self.action_dim, hidden_layers, activation=activation, duel=duel)
        self.dqn_target = DQN(self.state_dim, self.action_dim, hidden_layers, activation=activation, duel=duel)
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
        max_q, action = q.max(0)
        return action.data.cpu().numpy()

    def optimize(self, batch_size):
        batch, idxs, is_weights = self.memory.sample(batch_size)

        states = torch.from_numpy(np.array(batch["states"], dtype=np.float32))
        actions = torch.from_numpy(np.array(batch["actions"], dtype=np.int64))
        rewards = torch.from_numpy(np.array(batch["rewards"], dtype=np.float32))
        next_states = torch.from_numpy(np.array(batch["next_states"], dtype=np.float32))
        masks = torch.from_numpy(np.array(batch["masks"], dtype=np.float32))

        is_weights_t = None
        if is_weights is not None:
            is_weights_t = torch.from_numpy(is_weights).float()

        if use_gpu():
            states = states.cuda()
            actions = actions.cuda()
            rewards = rewards.cuda()
            next_states = next_states.cuda()
            masks = masks.cuda()
            if is_weights_t is not None:
                is_weights_t = is_weights_t.cuda()

        self.step(states, actions, rewards, next_states, masks, idxs, is_weights_t)

    def step(self, states, actions, rewards, next_states, masks, idxs, is_weights):
        state_action_values = self.dqn(states).gather(1, actions.unsqueeze(1))

        if self.double:
            pred_actions = self.dqn(next_states).max(1)[1]
            next_state_values = self.dqn_target(next_states).gather(1, pred_actions.unsqueeze(1)).view(-1).detach()
        else:
            next_state_values = self.dqn_target(next_states).max(1)[0].detach()
        pred_state_action_values = rewards + self.gamma * next_state_values * masks

        if not idxs is None and not is_weights is None:
            for i in range(len(states)):
                errors = torch.abs(state_action_values.view(-1) - pred_state_action_values).data.cpu().numpy()
                idx = idxs[i]
                self.memory.update(idx, errors[i])

            loss = self.loss_fct(state_action_values, pred_state_action_values.unsqueeze(1), reduction='none')
            loss *= is_weights.unsqueeze(1)
            loss = loss.mean()
        else:
            loss = self.loss_fct(state_action_values, pred_state_action_values.unsqueeze(1))

        # Backpropagation step
        self.optimizer.zero_grad()
        loss.backward()
        # for param in self.dqn.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def update_target(self):
        self.dqn_target.load_state_dict(self.dqn.state_dict())
