# -*- encoding:utf-8 -*- 
import math
import json
import pickle

import sys
from contextlib import closing
from io import StringIO
from gym import utils
from gym.envs.toy_text import discrete
import numpy as np


class PingPongEnvDis(discrete.DiscreteEnv):
    def __init__(self):
        self.b_height = 5
        self.b_width = 8
        self.rng = np.random.RandomState(0)
        num_states = self.b_height * self.b_width * 2 * 2 * self.b_height + 1
        num_actions = 3
        initial_state_distrib = np.zeros(num_states)
        P = {}
        P[0] = {}
        for action in range(num_actions):
            P[0][action] = [(1.0, 0, -1, 1)]
        for h in range(self.b_height):
            for w in range(self.b_width):
                for d1 in range(2):
                    for d2 in range(2):
                        for br in range(self.b_height):
                            ball = [h, w, d1, d2]
                            bar = [br, 0]
                            # print(ball,bar)
                            state = self.encode(ball, bar)
                            P[state] = {}
                            for action in range(num_actions):
                                P[state][action] = []

                                # print("current")
                                # print(ball,bar,action)
                                # print(self.get_state_str(ball,bar))

                                next_ball, next_bar, terminated, reward = self.gen_next_state(ball, bar, action)

                                # print("next")
                                # print('terminated',terminated)
                                # print(next_ball,next_bar,action)
                                # print(self.get_state_str(next_ball,next_bar))

                                new_state = self.encode(next_ball, next_bar)
                                P[state][action].append(
                                    (1.0, new_state, reward, terminated))
                            # print(ball,bar)
                            initial_state_distrib[state] = self.is_init_state(ball, bar)
        initial_state_distrib /= initial_state_distrib.sum()
        discrete.DiscreteEnv.__init__(
            self, num_states, num_actions, P, initial_state_distrib)

    def is_init_state(self, ball, bar):
        if ball[0] >= int(self.b_height / 2) - 1 and ball[0] <= int(self.b_height / 2) + 1 \
                and ball[1] >= self.b_width - 2 and ball[1] <= self.b_width - 1 \
                and bar[0] == int(self.b_height / 2):
            return 1
        else:
            return 0

    def is_init_state_old(self, ball, bar):
        if ball[0] == self.b_height // 2:
            if ball[1] == self.b_width - 1:
                if ball[2] == 0:
                    if ball[3] == 0:
                        if bar[0] == self.b_height // 2:
                            return 1
        return 0

    def encode(self, ball, bar):
        if ball[0] < 0:
            state = 0
        else:
            state = 0
            mult = 1
            state += bar[0] * mult
            mult *= self.b_height
            state += ball[3] * mult
            mult *= 2
            state += ball[2] * mult
            mult *= 2
            state += ball[1] * mult
            mult *= self.b_width
            state += ball[0] * mult
            state += 1
        return state

    def decode(self, state):
        ball = [0, 0, 0, 0]
        bar = [0, 0]
        if state == 0:
            ball = [-1, 0, 0, 0]
            bar[0] = int(self.b_height / 2)
        else:
            state -= 1
            bar[0] = state % self.b_height
            state = state // self.b_height
            ball[3] = state % 2
            state = state // 2
            ball[2] = state % 2
            state = state // 2
            ball[1] = state % self.b_width
            state = state // self.b_width
            ball[0] = state
        return ball, bar

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

    def gen_next_state(self, ball, bar, action):
        next_ball, next_bar = ball.copy(), bar.copy()
        self.move_bar_1(next_bar, action)
        terminated, reward = self.check_next_position(next_ball, next_bar)
        if terminated:
            next_ball = [-1, 0, 0, 0]
            next_bar = [int(self.b_height / 2),0]
        else:
            self.update_position(next_ball, next_bar)
        return next_ball, next_bar, terminated, reward

    def get_state_str(self, ball, bar):
        board_s = u""
        board_s += u"--------------------------\n"
        for i in range(self.b_height):
            if i == bar[0] and ball[0] >=0:
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
        ball, bar = self.decode(self.s)
        out_str = self.get_state_str(ball, bar)
        outfile.write(out_str)
        # No need to return anything for human
        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()


def main():
    PingPongEnvDis()
    # QLearningPingPong(is_curses=True,qname="qout_good2.json",epsilon=0,save_model=False)


if __name__ == "__main__":
    main()
