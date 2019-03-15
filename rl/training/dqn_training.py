import random

import numpy as np
import torch

from rl.agents.dqn_agent import DqnAgent
from rl.memory.memory import ReplayBuffer
from rl.util.logger import Logger


class DqnTraining:
    def __init__(self, env, seed=None, max_steps=1000, batch_size=32, mem_size=10000, learning_starts=1, learning_freq=1,
                 target_update=500, **agent_params):
        if not seed is None:
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.env = env
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.learning_freq = learning_freq
        self.target_update = target_update

        input_dim = self.env.observation_space.shape[0]
        output_dim = self.env.action_space.n
        self.dqn_agent = DqnAgent(input_dim, output_dim, **agent_params)
        self.memory = ReplayBuffer(capacity=mem_size)
        self.logger = Logger()

    def train(self):
        eps = 0
        eps_return = 0
        obs = self.env.reset()

        for t in range(self.max_steps):
            action = self.dqn_agent.act(obs, t)

            next_obs, reward, done, info = self.env.step(action)
            eps_return += reward

            self.memory.push(obs, action, reward, next_obs, 1 - done)

            if done:
                next_obs = self.env.reset()
                eps += 1
                self.logger.print_return(eps, eps_return)
                eps_return = 0
            obs = next_obs

            if len(self.memory) >= self.batch_size and t > self.learning_starts and t % self.learning_freq == 0:
                batch = self.memory.sample(self.batch_size)

                self.dqn_agent.optimize(batch)

            if t % self.target_update == 0:
                self.dqn_agent.update_target()