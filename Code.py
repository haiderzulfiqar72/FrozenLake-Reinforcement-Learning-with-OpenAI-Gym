import numpy as np
import matplotlib.pyplot as plt
import time
import gym

#enviroment creation
env = gym.make("FrozenLake-v1", is_slippery=False)
env.reset()
env.render()

state, reward, done, info = env.step(3)
env.render()

#Q(s,a) initialization
action_size = env.action_space.n
print("Action size: ", action_size)

state_size = env.observation_space.n
print("State size: ", state_size)

qtable = np.zeros((state_size, action_size))
print(qtable)

# Evaluation Policy
def eval_policy(qtable_, num_of_episodes_, max_steps_):
    rewards = []

    for episode in range(num_of_episodes_):
        state = env.reset()
        step = 0
        done = False
        total_rewards = 0

        for step in range(max_steps_):
            action = np.argmax(qtable_[state,:])
            new_state, reward, done, info = env.step(action)
            total_rewards += reward
        
            if done:
                rewards.append(total_rewards)
                break
            state = new_state
    avg_reward = sum(rewards)/num_of_episodes_
    return avg_reward

#Algorithm
for i in range(10):
    reward_best = -1000
    total_episodes = 400
    max_steps = 100
    gamma= 0.9
    rewards= []
    qtable = np.zeros((state_size, action_size))
    
    for episode in range(total_episodes):
        step = 0
        done = False
        if episode%10==0:
            avg_reward = eval_policy(qtable, 10, 100)
        state = env.reset()
    
        
        for step in range(max_steps):
            reward_tot = 0
            action = np.random.randint(0,4)   #0/left 1/down 2/right 3/up
            new_state, reward, done, info = env.step(action)
            qtable[state][action]= reward + gamma*max(qtable[new_state,:])
            # env.render()
            # print(qtable)
            # input()
            state= new_state
            
            if done == True or max_steps== True:
                if episode%10==0:
                    rewards.append(avg_reward)
                break  
        
    print(qtable)
    print(rewards)
    
    plt.plot(np.arange(0, total_episodes, 10), rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title('Deterministic Rule Version Performance')
    plt.grid(True)
    
plt.show()

#Part2
env = gym.make("FrozenLake-v1", is_slippery=True)
env.reset()

#Algorithm
for i in range(10):
    reward_best = -1000
    total_episodes = 1000
    max_steps = 100
    gamma= 0.9
    rewards= []
    qtable = np.zeros((state_size, action_size))
    
    for episode in range(total_episodes):
        step = 0
        done = False
        if episode%10==0:
            avg_reward = eval_policy(qtable, 10, 100)
        state = env.reset()
    
        
        for step in range(max_steps):
            reward_tot = 0
            action = np.random.randint(0,4)   #0/left 1/down 2/right 3/up
            new_state, reward, done, info = env.step(action)
            qtable[state][action]= reward + gamma*max(qtable[new_state,:])
            state= new_state
            
            if done == True or max_steps== True:
                if episode%10==0:
                    rewards.append(avg_reward)
                break  
        
    print(qtable)
    print(rewards)
    
    plt.plot(np.arange(0, total_episodes, 10), rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title('Non Deterministic Version Performance')
    plt.grid(True)
plt.show()

#Part3
env = gym.make("FrozenLake-v1", is_slippery=True)
env.reset()

#Algorithm
for i in range(10):
    reward_best = -1000
    total_episodes = 1000
    max_steps = 100
    gamma= 0.9
    alpha= 0.1
    rewards= []
    qtable = np.zeros((state_size, action_size))
    
    for episode in range(total_episodes):
        step = 0
        done = False
        if episode%10==0:
            avg_reward = eval_policy(qtable, 10, 100)
        state = env.reset()
    
        
        for step in range(max_steps):
            reward_tot = 0
            action = np.random.randint(0,4)   #0/left 1/down 2/right 3/up
            new_state, reward, done, info = env.step(action)
            qtable[state][action]= qtable[state][action] + alpha*(reward + gamma * max(qtable[new_state,:]) - qtable[state][action]) 
            state= new_state
            
            if done == True or max_steps== True:
                if episode%10==0:
                    rewards.append(avg_reward)
                break  
        
    print(qtable)
    print(rewards)
    
    plt.plot(np.arange(0, total_episodes, 10), rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title('Updated Non Deterministic Rule Performance')
    plt.grid(True)
plt.show()
