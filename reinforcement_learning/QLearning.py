import gym
import gym_walk
import numpy as np

class QLearningAgent(object):
    def __init__(self, env_name):
        self.Q = None
        self.alpha = 0.2  # param
        self.gamma = 0.90
        self.epsilon = 0.25 # param
        self.n_episodes = 1_000 # param
        self.env = gym.make(env_name).env

    def solve(self):
        """Create the Q table"""
        self.Q = np.zeros([self.env.observation_space.n, self.env.action_space.n])
        for n in range(self.n_episodes):
            # Initialize S
            s = self.env.reset()
            done = False
            
            # Loop for each step of episode
            while not done:
                # Choose A from S
                if np.random.rand() <= self.epsilon:
                    a = np.random.randint(self.env.action_space.n)
                else:
                    a = np.argmax(self.Q[s, :])
                
                # Take action A
                s_prime, reward, done, info = self.env.step(a)
                
                # Update Q Table
                self.Q[s, a] += self.alpha * (reward + self.gamma * self.Q[s_prime, np.argmax(self.Q[s_prime, :])] - self.Q[s, a])
                s = s_prime
        return reward

    def Q_table(self, state, action):
        """return the optimal value for State-Action pair in the Q Table"""
        return self.Q[state][action]
