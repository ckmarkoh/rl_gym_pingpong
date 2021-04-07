import numpy as np
import pickle

import logging
import logging.handlers
logger = logging.getLogger("")
logger.setLevel(logging.DEBUG)
handler = logging.handlers.RotatingFileHandler(
    "out_t_qlearn.log", maxBytes=(1048576*5), backupCount=7
)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

class QLearningTrainer(object):
    def __init__(self, enviroment):
        self.enviroment = enviroment
        self.q_table = np.zeros([self.enviroment.observation_space.n, self.enviroment.action_space.n])
        self.rng = np.random.RandomState(0)

    def save_model(self):
        pickle.dump(self.q_table, open("model/q_learning.pkl", "wb"))

    def load_model(self):
        self.q_table = pickle.load(open("model/q_learning.pkl", "rb"))

    def train(self):
        alpha = 0.1
        gamma = 0.6
        epsilon = 0.1

        reward_list = []
        pos_reward = 0
        num_of_episodes = 10000
        # num_of_episodes = 10000

        for episode in range(0, num_of_episodes):
            state = self.enviroment.reset()
            terminated = False
            while not terminated:
                if np.random.rand() > epsilon:
                    new_action_states = [(x,i)for i,x in enumerate(self.q_table[state])]
                    np.random.shuffle(new_action_states)
                    action = max(new_action_states,key=lambda x:x[0])[1]
                    action = action
                    #action = np.argmax(self.q_table[state])
                else:
                    action = np.random.choice(range(self.enviroment.action_space.n))

                new_state, new_reward, terminated, info = self.enviroment.step(action)
                if new_reward != 0:
                    reward_list.append(new_reward)

                new_qa = np.max(self.q_table[new_state])
                current_qa = self.q_table[state,action]
                #logging.info("pre",self.q_table[state,action])
                self.q_table[state,action] += alpha*(new_reward+gamma*new_qa-current_qa)
                #logging.info("next",self.q_table[state,action])
                state = new_state


            if (episode+ 1) % 100 == 0:
                # clear_output(wait=True)
                logging.info("Episode: {}".format(episode + 1))
                logging.info("Reward: {}".format(np.average(reward_list) if len(reward_list) >0 else 0   ))
                reward_list.clear()
                self.enviroment.render()

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
                action = np.argmax(self.q_table[state])
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
