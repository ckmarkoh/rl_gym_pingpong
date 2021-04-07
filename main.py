import numpy as np
import argparse
import random

import gym
from t_qlearning import QLearningTrainer
from t_dqn import DQNTrainer
from t_policy_gradient import PolicyGradientTrainer
from env_pingpong import PingPongEnv
from env_pingpong_dis import PingPongEnvDis


def main(args):
    #enviroment = gym.make("Taxi-v3").env
    enviroment = PingPongEnv()
    enviroment.render()
    q = PolicyGradientTrainer(enviroment)
    if not args.eval:
        q.train()
        q.eval()
    else:
        q.eval()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval",action='store_true')
    parser.add_argument("--vis",action='store_true')
    args = parser.parse_args()
    main(args)