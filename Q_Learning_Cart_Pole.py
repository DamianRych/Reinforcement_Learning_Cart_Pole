import random
import numpy as np
import gym 
import math
import matplotlib.pyplot as plt

#state ranges
#x e [-4.8, 4.8]
#v e [-3.4*10^38, 3.4*10^38]
#t e [-0.42, 0.42]
#w e [-3.4*10^38, 3.4*10^38]

class q_table_learning():
    def __init__(self, lr_min, exp_rate_min, discount, action_space):
        self.lr = 0
        self.exp_rate = 0
        self.action_space = action_space
        self.lr_min = lr_min
        self.exp_rate_min = exp_rate_min
        self.discount = discount
        self.buckets = (64, 64, 64, 144,)
        self.q_table = self.create_q_table()
        


    def create_q_table(self):
        #creating buckets for makeing dicrete states
        #the less buckets a state has the bigger the loss of information
        q_table = np.zeros(self.buckets + (self.action_space,))
        return q_table

    def get_lr(self, episode):
        self.lr = max(self.lr_min, min(1.0, 1.0 - math.log10((episode + 1)/5000)))

    def choose_action(self, discrete_state):
    #we need to take a look into the q_table and take the best value or explore the enviroment randomly
        if(random.uniform(0,1) < self.exp_rate):
            action = np.random.randint(0,2)
            return action
        else:
            action = np.argmax(self.q_table[discrete_state])
            return action    

    def get_exploration_rate(self, episode):
        self.exp_rate = max(self.exp_rate_min, min(1.0, 1.0 - math.log10((episode + 1)/5000)))   


    def make_discrete(self, new_state, env):
        upper_bounds = [env.observation_space.high[0], 0.5, env.observation_space.high[2], math.radians(50)]
        lower_bounds = [env.observation_space.low[0], -0.5, env.observation_space.low[2], -math.radians(50)]
        ratios = [(new_state[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(new_state))]
        #turns the ratio values into buckets  
        discrete_state = [int(round((self.buckets[i] - 1) * ratios[i])) for i in range(len(new_state))]
        
        #makes sure that we stay in the range of the buckets 
        #we either return a number that that is lower then max bucket number otherwise we return the max bucket number
        discrete_state = [min(self.buckets[i]- 1, max(0, discrete_state[i])) for i in range(len(new_state))]
        #return 4 values which are grouped in different buckets
        return tuple(discrete_state)

    def update_q_value(self,old_discrete_state, new_discrete_state, reward, action, done):
        if done:
            self.q_table[old_discrete_state][action] += reward
        else:
            self.q_table[old_discrete_state][action] += self.lr * (reward + self.discount * (np.max(self.q_table[new_discrete_state])- self.q_table[old_discrete_state][action]))

    


class MakeGame():
    def __init__(self, epochs, render, learn_constant, batch_size):
        self.epochs = epochs
        self.render = render
        #how fast is the transition from exploration to exploitation
        self.learn_constant = learn_constant
        self.env = env = gym.make('CartPole-v1')
        self.best_score = 0
        self.avg_score = 0
        self.count = 0
        self.batch_size = batch_size
        #for plotting
        self.x_values = []
        self.y_values = []
        #increase max step for an episode default is 500 steps
        self.env._max_episode_steps = 4_000_000

    def get_action_space(self):
        return self.env.action_space.n

    def get_env(self):
        return self.env


    def play(self, q_table): 
        self.x_values = []
        self.y_values = []
        for episode in range(self.epochs):
            observation = self.env.reset()
            discrete_state = q_table.make_discrete(observation, self.env)
            t = 0
            done = False
            render_count = False
            lr = q_table.get_lr(episode)
            explore_chance = q_table.get_exploration_rate(episode) 
            #batch capacity reached
            if(self.count == self.batch_size):
                print(f'Average batch score: {int(self.avg_score/self.batch_size)} best batch score: {self.best_score} current epoch {episode}')
                self.x_values.append(episode)
                self.y_values.append(int(self.avg_score/self.batch_size))
                self.best_score = 0
                self.avg_score = 0
                self.count = 0
                render_count = True
            self.count += 1
            while not done:
                t += 1
                if(self.render and render_count):
                    self.env.render()
                #print(observation)
                action = q_table.choose_action(discrete_state) 
                new_state, reward, done, info = self.env.step(action)
                new_discrete_state = q_table.make_discrete(new_state, self.env)
                reward = 1 if not done else -1
                q_table.update_q_value(discrete_state, new_discrete_state, reward, action, done)
                if done:
                    if(self.best_score < t):
                        self.best_score = t
                    self.avg_score += t
                    break
                else:
                    discrete_state = new_discrete_state
        self.env.close()
        plt.plot(self.x_values, self.y_values)
        plt.ylabel('Score')
        plt.xlabel('Episode')
        plt.show()


class main():
    epcohs = 5_000
    render = False
    batch_size = 1000
    exp_rate_min = 0.01 
    lr_min = 0.01
    discount = 0.9
    #how fast is the transition from exploration to exploitation
    learn_constant = 5000
    game = MakeGame(epcohs, render, learn_constant, batch_size)
    q_table = q_table_learning(lr_min, exp_rate_min, discount, game.get_action_space())
    game.play(q_table)

if __name__ == '__main':
    main()
