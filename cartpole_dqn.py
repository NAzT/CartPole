
from __future__ import print_function

import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class Agent:

    def __init__(self, state_size, action_size, batch_size):
        self.state_size    = state_size
        self.action_size   = action_size
        self.batch_size    = batch_size
        self.memory        = deque(maxlen=2000)
        self.gamma         = 0.95   # discount rate
        self.epsilon       = 1.0    # exploration rate
        self.epsilon_min   = 0.01
        self.epsilon_decay = 0.995
        self.dqn           = self.build_model()
        
    def build_model(self):
        dqn = Sequential()
        dqn.add(Dense(24, input_dim=self.state_size, activation='tanh')) # input dimension = #states
        dqn.add(Dense(self.action_size, activation='linear'))            # output nodes = #action
        dqn.compile(loss='mse', optimizer=Adam(lr=0.01))                      
        print(dqn.summary())
        return dqn

    def act(self, state, explore):
        if explore and np.random.rand() <= self.epsilon: # explore/exploit tradeoff
            return random.randrange(self.action_size)
        act_values = self.dqn.predict(state)
        return np.argmax(act_values[0]) 

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.memory)<self.batch_size:
            return

        X, dqnY = [], []
        minibatch = random.sample(self.memory, self.batch_size) 
        for state, action, reward, next_state, done in minibatch:
            X.append( state[0] )
            target = reward if done else reward + self.gamma * np.max(self.dqn.predict(next_state)[0])
            target_dqn = self.dqn.predict(state)
            target_dqn[0][action] = target
            dqnY.append( target_dqn[0] )

        self.dqn.train_on_batch( np.array(X), np.array(dqnY) )

        if self.epsilon > self.epsilon_min:    # gradually change from explore to exploit
            self.epsilon *= self.epsilon_decay


if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    print("{} actions, {}-dim state".format(action_size, state_size))

    agent = Agent(state_size, action_size, 32)
    
    emax = 5000
    for e in range(emax):
        state = env.reset()
        state = state.reshape((1, state_size))
        for time in range(200):
            env.render()
            action = agent.act(state, True)
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.reshape( (1,state_size) )
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}".format(e, emax, time, agent.epsilon))
                break
            
        agent.train()

        for t in range(100):
            success = False
            state = env.reset()
            state = state.reshape((1, state_size))
            for time in range(200):
                #env.render()
                action = agent.act(state, False)
                next_state, reward, done, _ = env.step(action)
                next_state = next_state.reshape( (1,state_size) )
                state = next_state
                
                if time>195:
                    success = True
                    break
                if done:
                    break
            if not success:
                break
        if success:
            print("Cartpole-v0 SOLVED!!! {}".format(e))
            break
            
            
