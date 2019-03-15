import unittest

import gym
import torch

from rl.training.dqn_training import DqnTraining

torch.set_default_tensor_type(torch.cuda.FloatTensor)


class TestDqn(unittest.TestCase):

    def test_dqn(self):
        seed = 123
        env = gym.make("CartPole-v0")
        env.seed(seed)

        params = dict(seed=seed, max_steps=10000, batch_size=32, mem_size=10000, learning_starts=1, learning_freq=1,
                      target_update=500, lr=1e-3, l2_reg=1e-3, hidden_layers=[32, 32], gamma=1.0, eps_decay=.1,
                      eps_init=1., eps_min=0.02)

        training = DqnTraining(env, **params)
        training.train()

        training.logger.plot_return()


if __name__ == '__main__':
    unittest.main()
