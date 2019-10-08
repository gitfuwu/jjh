import gym
import numpy as np

env = gym.make('MountainCar-v0')
env = env.unwrapped
episodes = 1000
for episode in range(episodes):
    print("episode: ", episode)
    state = env.reset()
    total_reward = 0
    while True:
        action = np.random.randint(0, 3)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        # print("action: ", action)
        # print("reward: ", reward)
        # print("done: ", done)

        # if done:
        #     reward = 10
        # agent.store_transition(state, action, reward, next_state, done)
        # agent.learn()
        state = next_state
        # print("step: ", step)
        # print("reward: ", reward)
        if done:
            print("total_reward: ", total_reward)
            break
