import numpy as np
import random
import torch
from torch.autograd import Variable
import pickle


import logging
import logging.handlers
logger = logging.getLogger("")
logger.setLevel(logging.DEBUG)
handler = logging.handlers.RotatingFileHandler(
    "out_t_dqn.log", maxBytes=(1048576*5), backupCount=7
)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class DQN():
    ''' Deep Q Neural Network class. '''

    def __init__(self, state_dim, action_dim, hidden_dim=16, lr=0.002):
        self.criterion = torch.nn.MSELoss()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, action_dim)
        )
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr)
        self.replay_size = 20

    def update(self, state, y, action):
        """Update the weights of the network given a training sample. """
        y_pred = self.model(torch.Tensor(state))
        loss = self.criterion(y_pred, Variable(torch.Tensor(y)))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def predict(self, state):
        """ Compute Q values for all actions using the DQL. """
        with torch.no_grad():
            return self.model(torch.Tensor(state))

    def replay_memory(self, memory, gamma=0.9):
        """ Add experience replay to the DQN network class. """
        # Make sure the memory is big enough
        if len(memory) >= self.replay_size:
            states = []
            targets = []
            # Sample a batch of experiences from the agent's memory
            batch = random.sample(memory, self.replay_size)
            # Extract information from the data
            for state, action, next_state, reward, done in batch:
                states.append(state)
                # Predict q_values
                q_values = self.predict(state).tolist()
                if done:
                    q_values[action] = reward
                else:
                    q_values_next = self.predict(next_state)
                    q_values[action] = reward + gamma * torch.max(q_values_next).item()
                targets.append(q_values)
                self.update(states, targets, action)


class DQNTrainer(object):
    def __init__(self, enviroment):
        self.enviroment = enviroment
        self.model = DQN(self.enviroment.observation_space.nvec,self.enviroment.action_space.n)
        self.rng = np.random.RandomState(0)


    def save_model(self):
        torch.save(self.model, open("model/dqn_16.th", "wb"))

    def load_model(self):
        self.model = torch.load(open("model/dqn_16.th", "rb"))


    def train(self):
        #alpha = 0.1
        gamma = 0.9
        epsilon = 0.3
        eps_decay = 0.99


        memory = []

        reward_list = []
        pos_reward = 0
        num_of_episodes = 3000
        # num_of_episodes = 10000

        for episode in range(0, num_of_episodes):
            state = self.enviroment.reset()

            reward = 0
            terminated = False

            while not terminated:
                if self.rng.uniform(0, 1) < epsilon:
                    action = self.enviroment.action_space.sample()
                else:
                    q_values = self.model.predict(state)
                    action = torch.argmax(q_values).item()

                next_state, reward, terminated, info = self.enviroment.step(action)
                memory.append((state, action, next_state, reward, terminated))
                self.model.replay_memory(memory, gamma=gamma)
                if reward != 0:
                    reward_list.append(reward)

                state = next_state

            if (episode + 1) % 100 == 0:
                logging.info("Episode: {}".format(episode + 1))
                logging.info("Reward: {}".format(np.average(reward_list) if len(reward_list) >0 else 0   ))
                reward_list.clear()
                self.enviroment.render()
                self.save_model()

            if len(memory) > 500:
                memory = memory[-500:]
            epsilon = max(epsilon * eps_decay, 0.01)

        logging.info("**********************************")
        logging.info("Training is done!\n")
        logging.info("**********************************")

        self.save_model()

    def eval(self, vis=False):
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
                q_values = self.model.predict(state)
                action = torch.argmax(q_values).item()
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

        logging.info("**********************************")
        logging.info("Results")
        logging.info("**********************************")
        logging.info("Epochs per episode: {}".format(total_epochs / num_of_episodes))
        logging.info("Penalties per episode: {}".format(total_penalties / num_of_episodes))
        logging.info("Reward: {}".format(np.average(reward_list) if len(reward_list) > 0 else 0))
