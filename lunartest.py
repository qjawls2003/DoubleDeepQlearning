import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib
import matplotlib.pyplot as plt
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
config = tf.compat.v1.ConfigProto(gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.5))
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)
tf.compat.v1.disable_eager_execution()
output_dir = 'lunarOutput/lunar'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=100000)
        self.gamma = 0.99    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.0001
        self.model = self._build_model()
        self.minReward = -1500
        self.trainIt = 1
        self.startdecay = 0

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(256, input_dim=self.state_size, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action


    def load(self,name):
        self.model.load_weights(name)

    def save(self,name):
        self.model.save_weights(name)


def test(filename):

    episodes = 100

    env = gym.make('LunarLander-v2')

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size,action_size)
    agent.load(filename)
    env.seed(0)
    np.random.seed(0)
    agent.epsilon = 0
    for e in range(1, episodes+1):
        done = False
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        total_reward = 0.0

        while not done:

            env.render()

            action = agent.act(state)

            next_state, reward, done, _ = env.step(action)

            total_reward += reward
            next_state = np.reshape(next_state, [1, state_size])
            state = next_state

            if done:
                # print the score and break out of the loop
                print("episode: {}/{}, score: {}".format(e, episodes, total_reward))
                break

if __name__ == "__main__":

    test('lunarOutput/lunar/Figure2/lunarweights_0900.hd5f')
