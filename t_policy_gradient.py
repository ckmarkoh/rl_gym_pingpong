import numpy as np
import random
import torch
from torch.autograd import Variable
from torch.distributions import Bernoulli,Categorical

import pickle

import logging
import logging.handlers
logger = logging.getLogger("")
logger.setLevel(logging.DEBUG)
handler = logging.handlers.RotatingFileHandler(
    "out_t_policy_gradient.log", maxBytes=(1048576*5), backupCount=7
)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class Network():
    ''' Deep Q Neural Network class. '''

    def __init__(self, state_dim, action_dim, hidden_dim=16, lr=0.002):
        self.criterion = torch.nn.MSELoss()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, action_dim),
            torch.nn.Softmax()
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)

    def update(self, states, actions, rewards):
        """Update the weights of the network given a training sample. """
        self.optimizer.zero_grad()
        for state,action,reward in zip(states,actions,rewards):
            state = Variable(torch.Tensor(state))
            action = Variable(torch.Tensor([action]))
            probs = self.model(state)
            m = Categorical(probs)
            loss = -m.log_prob(action)*reward
            loss.backward()
        self.optimizer.step()

    def predict(self, state):
        """ Compute Q values for all actions using the DQL. """
        with torch.no_grad():
            return self.model(torch.Tensor(state))



class PolicyGradientTrainer(object):
    def __init__(self, enviroment):
        self.enviroment = enviroment
        self.model = Network(self.enviroment.observation_space.nvec,self.enviroment.action_space.n)
        self.rng = np.random.RandomState(0)


    def save_model(self):
        torch.save(self.model, open("model/policy_gradient_16.th", "wb"))

    def load_model(self):
        self.model = torch.load(open("model/policy_gradient_16.th", "rb"))


    def train(self):
        #alpha = 0.1
        gamma = 0.9
        epsilon = 0.3
        eps_decay = 0.99



        episode_state = []
        episode_action = []
        episode_reward = []

        reward_list = []
        pos_reward = 0
        num_of_episodes = 50000
        # num_of_episodes = 10000

        for episode in range(0, num_of_episodes):
            state = self.enviroment.reset()

            terminated = False


            while not terminated:
                #if self.rng.uniform(0, 1) < epsilon:
                probs = self.model.predict(state)
                action = Categorical(probs).sample()
                action = int(action.data.numpy())
                #else:
                #    q_values = self.model.predict(state)
                #action = torch.argmax(q_values).item()

                next_state, reward, terminated, info = self.enviroment.step(action)
                episode_state.append(state)
                episode_action.append(action)
                episode_reward.append(reward)

                if reward != 0:
                    reward_list.append(reward)
                state = next_state

            for i in reversed(range(len(episode_reward))):
                if i != len(episode_reward) -1:
                    episode_reward[i] = (1-gamma)*episode_reward[i] + gamma*episode_reward[i+1]

            self.model.update(episode_state,episode_action,episode_reward)
            episode_state.clear()
            episode_action.clear()
            episode_reward.clear()


            if (episode + 1) % 100 == 0:
                logger.info("Episode: {}".format(episode + 1))
                logger.info("Reward: {}".format(np.average(reward_list) if len(reward_list) >0 else 0   ))
                reward_list.clear()
                self.enviroment.render()
                self.save_model()

            #if len(memory) > 00:
            #    memory = memory[-500:]
            epsilon = max(epsilon * eps_decay, 0.01)

        logger.info("**********************************")
        logger.info("Training is done!\n")
        logger.info("**********************************")

        self.save_model()

    def eval(self, vis=True):
        self.load_model()
        total_epochs = 0
        total_penalties = 0
        num_of_episodes = 100

        reward_list = [0]

        for _ in range(num_of_episodes):
            state = self.enviroment.reset()
            epochs = 0
            penalties = 0
            reward = 0

            terminated = False
            if vis:
                self.enviroment.render()

            while not terminated:
                probs = self.model.predict(state)
                action = Categorical(probs).sample()
                action = int(action.data.numpy())

                state, reward, terminated, info = self.enviroment.step(action)
                if reward != 0:
                    reward_list.append(reward)
                if vis:
                    self.enviroment.render()

                if reward == -10:
                    penalties += 1

                epochs += 1
                if epochs > 100:
                    break
            total_penalties += penalties
            total_epochs += epochs

        logger.info("**********************************")
        logger.info("Results")
        logger.info("**********************************")
        logger.info("Epochs per episode: {}".format(total_epochs / num_of_episodes))
        logger.info("Penalties per episode: {}".format(total_penalties / num_of_episodes))
        logger.info("Reward: {}".format(np.average(reward_list) if len(reward_list) > 0 else 0))
