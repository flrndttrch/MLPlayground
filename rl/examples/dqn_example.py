import argparse

import gym
import torch
import torch.nn.functional as F

from rl.training.dqn_training import DqnTraining
from rl.util.logger import Logger

torch.set_default_tensor_type(torch.cuda.FloatTensor) if torch.cuda.is_available() else torch.set_default_tensor_type(
    torch.FloatTensor)


def run_dqn_example(env_str):
    print("\nRunning DQN on {}".format(env_str))
    seed = 123
    env = gym.make(env_str)
    env.seed(seed)

    params = dict(seed=seed, max_steps=10000, batch_size=32, mem_size=10000, learning_starts=1, learning_freq=1,
                  target_update=500, lr=1e-3, l2_reg=1e-3, hidden_layers=[32, 32], gamma=1.0, double=False,
                  duel=False, mem_type='per', loss_fct=F.mse_loss, eps_decay = .0002, eps_init = 1., eps_min = 0.02)

    Logger.print_params(params)

    training = DqnTraining(env, **params)
    training.train()

    training.logger.plot_return("DQN on {}".format(env_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", dest="env", type=str, help="String defining the environment according to OpenAI gym.")
    args = parser.parse_args()

    run_dqn_example(args.env)
