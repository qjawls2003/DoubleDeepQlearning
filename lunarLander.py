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
        self.gamma = 0.99    #0.99
        self.epsilon = 1.0  # 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99 #0.99
        self.learning_rate = 0.0001 #0.0001
        self.model = self.model()
        self.minReward = -1500
        self.trainIt = 1
        self.startdecay = 0

    def model(self):
        model = Sequential()
        model.add(Dense(256, input_dim=self.state_size, activation='relu')) #256
        model.add(Dense(128, activation='relu')) #128
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        thisbatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in thisbatch:
            target = reward
            if not done:
              target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target2 = self.model.predict(state)
            target2[0][action] = target
            history = self.model.fit(state, target2, epochs=1, verbose=0)

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

        return history

    def load(self,name):
        self.model.load_weights(name)

    def save(self,name):
        self.model.save_weights(name)

def train():
    env = gym.make('LunarLander-v2')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size,action_size)
    env.seed(0)
    np.random.seed(0)
    batch_size = 10
    episodes = 2000
    reward_per_eps = []
    reward_per_100 = []
    epsilonList = []
    reward_100 = 0
    counter = 0
    avgReward = 0
    lossList = []
    steps = 0

    for e in range(1, episodes+1):
        done = False
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        total_reward = 0.0
        #for t in range(1000):
        for t in range(10000):
            # turn this on if you want to render every 25 episodes
            if e % 25 == 0:
                #env.render()
                continue

            action = agent.act(state)
            next_state, reward, done, extra = env.step(action)
            total_reward += reward
            next_state = np.reshape(next_state, [1, state_size])
            agent.memorize(state, action, reward, next_state, done)
            state = next_state

            if done or total_reward < agent.minReward:
                # print the score and break out of the loop
                print("episode: {}/{}, score: {} e: {}".format(e, episodes, total_reward, agent.epsilon))
                break

            steps += 1

            if len(agent.memory) > batch_size and steps % agent.trainIt == 0:
                hist = agent.replay(batch_size)
                loss = hist.history['loss'][0]
                lossList.append(loss)

            if agent.epsilon > agent.epsilon_min and agent.startdecay > steps:
                agent.epsilon -= 0.98/agent.startdecay
                #linear epsilon_decay

        if agent.epsilon > agent.epsilon_min and agent.startdecay < steps:
            agent.epsilon *= agent.epsilon_decay
            #regular epsilon_decay

        epsilonList.append(agent.epsilon)
        reward_per_eps.append(total_reward)
        reward_100 += total_reward
        if e % 100 == 0:
            agent.save(output_dir + "weights_" + '{:04d}'.format(e) + ".hd5f")
            avgReward = reward_100/100
            reward_per_100.append(avgReward)
            reward_100 = 0
            print("Average Return at Episode {}: {}".format(e, avgReward))
            if avgReward > 200:
                counter += 1

        if counter > 2: #received an average score > 200 three times
            break

    return [reward_per_eps,reward_per_100,lossList, epsilonList]
''' #in lunartest.py
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
'''
def plot(iterations,per100,lossIt,epIt, reward_per_eps, reward_per_100, lossList, epList):
    fig, axs = plt.subplots(2,2)
    axs[0,0].plot(iterations, reward_per_eps, 'tab:blue')
    axs[0,1].plot(per100, reward_per_100, 'tab:green')
    axs[1,0].plot(lossIt, lossList, 'tab:red')
    axs[1,1].plot(epIt, epList, 'tab:orange')
    axs[0,0].set_title('Reward per Episode')
    axs[0,1].set_title('Average Reward per 100 Episodes')
    axs[1,0].set_title('Loss per Step')
    axs[1,1].set_title('Epsilon per Episode')
    axs[0,0].set(xlabel='Episodes', ylabel='Rewards')
    axs[0,1].set(xlabel='Episodes', ylabel='Average Rewards')
    axs[1,0].set(xlabel='Steps', ylabel='Loss')
    axs[1,1].set(xlabel='Episodes', ylabel='Epsilon')
    plt.show()

if __name__ == "__main__":

    #test('lunarOutput/lunarweights_0400.hd5f')
    p = train()
    iterations = range(len(p[0]))
    iterations_100 = range(100,(len(p[1])*100)+1,100)
    iterations_loss = range(len(p[2]))
    iterations_ep = range(len(p[3]))
    plot(iterations,iterations_100,iterations_loss,iterations_ep, p[0],p[1],p[2],p[3])
