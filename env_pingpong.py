# -*- encoding:utf-8 -*- 
import math
import json
import pickle

import sys
from contextlib import closing
from io import StringIO
from gym import utils
from gym.envs.toy_text import discrete
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np


class PingPongEnv(gym.Env):
    def __init__(self):
        self.b_height = 5
        self.b_width = 8
        self.rng = np.random.RandomState(0)
        self.ball = [int(self.b_height / 2), self.b_width - 2, 0, 0]
        self.bar = [int(self.b_height / 2), 1]

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.MultiDiscrete(5)
        self.rng = np.random.RandomState(1)

    def reset(self):
        self.ball = [int(self.b_height / 2) + self.rng.choice([-1, 0, 1]),
                     self.b_width + self.rng.choice([-3, -2,-1]),
                     self.rng.choice([0, 1]),
                     self.rng.choice([0, 1]),
                     ]
        self.bar = [int(self.b_height / 2), 1]
        self.state = [self.ball[0], self.ball[1], self.ball[2], self.ball[3], self.bar[0]]
        return np.array(self.state)

    def step(self, action):
        # next_ball, next_bar = ball.copy(), bar.copy()
        self.move_bar_1(self.bar, action)
        terminated, reward = self.check_next_position(self.ball, self.bar)
        if terminated:
            self.ball = [-1, 0, 0, 0]
            self.bar = [int(self.b_height / 2), 0]
        else:
            self.update_position(self.ball, self.bar)

        self.state = [self.ball[0], self.ball[1], self.ball[2], self.ball[3], self.bar[0]]
        return np.array(self.state), reward, terminated, {}

    def check_next_position(self, ball, bar):
        terminated = 0
        reward = 0

        if ball[1] == 0:
            if ball[0] == bar[0]:
                ball[3] = 1
                reward = 1
            else:
                terminated = 1
                reward = -1

        if ball[1] == self.b_width - 1:
            ball[3] = 0

        if ball[0] == 0:
            ball[2] = 1
        elif ball[0] == self.b_height - 1:
            ball[2] = 0

        if bar[0] <= 0 and bar[1] == 0:
            bar[1] = 1
        elif bar[0] >= self.b_height - 1 and bar[1] == 2:
            bar[1] = 1

        return terminated, reward

    def move_bar_1(self, bar, act):
        bar[1] = act

    def update_position(self, ball, bar):
        ball[0] += -1 + 2 * ball[2]
        ball[1] += -1 + 2 * ball[3]
        bar[0] += -1 + bar[1]
        assert ball[0] >= 0 and ball[0] <= self.b_height - 1
        assert ball[1] >= 0 and ball[1] <= self.b_width - 1

    def get_state_str(self, ball, bar):
        board_s = u""
        board_s += u"--------------------------\n"
        for i in range(self.b_height):
            if i == bar[0] and ball[0] >= 0:
                board_s += u"■ "
            else:
                board_s += u"☐ "
            for j in range(self.b_width):
                if i == ball[0] and j == ball[1]:
                    board_s += u"⬤"
                else:
                    board_s += u"☐"
            board_s += u"\n"
        board_s += u"--------------------------\n"
        board_s += u"\n"
        return board_s

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        ball, bar = self.ball,self.bar
        out_str = self.get_state_str(ball, bar)
        outfile.write(out_str)
        # No need to return anything for human
        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()


def main():
    PingPongEnv()
    # QLearningPingPong(is_curses=True,qname="qout_good2.json",epsilon=0,save_model=False)


if __name__ == "__main__":
    main()
