import unittest

import gym
import torch

from rl.training.dqn_training import DqnTraining

torch.set_default_tensor_type(torch.cuda.FloatTensor)


class TestDqn(unittest.TestCase):
    def test_dqn(self):
        env_str = "CartPole-v0"
        print("\nRunning vanilla DQN on {}".format(env_str))
        seed = 123
        env = gym.make(env_str)
        env.seed(seed)

        params = dict(seed=seed, max_steps=100, double=False, duel=False)

        training = DqnTraining(env, **params)
        training.train()

    def test_double_dqn(self):
        env_str = "CartPole-v0"
        print("\nRunning DQN with double update on {}".format(env_str))
        seed = 123
        env = gym.make(env_str)
        env.seed(seed)

        params = dict(seed=seed, max_steps=100, double=True, duel=False)

        training = DqnTraining(env, **params)
        training.train()

    def test_duel_dqn(self):
        env_str = "CartPole-v0"
        print("\nRunning DQN with duel architecture on {}".format(env_str))
        seed = 123
        env = gym.make(env_str)
        env.seed(seed)

        params = dict(seed=seed, max_steps=100, double=False, duel=True)

        training = DqnTraining(env, **params)
        training.train()


if __name__ == '__main__':
    unittest.main()
