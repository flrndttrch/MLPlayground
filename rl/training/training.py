import random

import numpy as np
import torch

from rl import use_gpu
from rl.agents.dqn_agent import DqnAgent
from rl.util.logger import Logger


class Training:
    def __init__(self, env, agent_cls, seed=None, max_steps=1000, batch_size=32, **kwargs):
        if not seed is None:
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.env = env
        self.max_steps = max_steps
        self.batch_size = batch_size
        input_dim = self.env.observation_space.shape[0]
        output_dim = self.env.action_space.n
        self.agent = agent_cls(input_dim, output_dim, **kwargs)
        self.logger = Logger()

    def train(self):
        pass


class OffPolicyTraining(Training):
    def __init__(self, env, seed=None, max_steps=1000, batch_size=32, learning_starts=1, learning_freq=1,
                 target_update=500, **agent_params):
        super(OffPolicyTraining, self).__init__(env, DqnAgent, seed=seed, max_steps=max_steps, batch_size=batch_size,
                                                **agent_params)

        self.learning_starts = learning_starts
        self.learning_freq = learning_freq
        self.target_update = target_update

    def train(self):
        eps = 0
        eps_return = 0
        obs = self.env.reset()

        for t in range(self.max_steps):
            action = self.agent.act(obs, t)

            next_obs, reward, done, info = self.env.step(action)
            eps_return += reward

            error = None
            if self.agent.mem_type is 'per':
                obs_t = torch.from_numpy(obs).float()
                action_t = torch.from_numpy(np.array([action])).long()
                next_obs_t = torch.from_numpy(next_obs).float()

                if use_gpu():
                    obs_t, action_t, next_obs_t = obs_t.cuda(), action_t.cuda(), next_obs_t.cuda()

                value = self.agent.dqn(obs_t).gather(0, action_t).item()
                target_val = self.agent.dqn_target(next_obs_t).max(0)[1].item()

                error = abs(value - target_val)

            self.agent.memory.push(obs, action, reward, next_obs, 1 - done, error=error)

            if done:
                next_obs = self.env.reset()
                eps += 1
                self.logger.print_return(eps, eps_return)
                eps_return = 0
            obs = next_obs

            if len(
                    self.agent.memory) >= self.batch_size and t > self.learning_starts and t % self.learning_freq == 0:
                self.agent.optimize(self.batch_size)

            if t % self.target_update == 0:
                self.agent.update_target()
