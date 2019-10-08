import gym
from NatureDQN import NatureDQN

env_name = 'Breakout-ram-v0'
episodes = 3000
steps = 300
test = 10

lr = 0.001
epsilon = 0.1
replay_size = 100
gamma = 0.9
batch_size = 32
env = gym.make(env_name)
agent = NatureDQN(env, lr, epsilon, replay_size, gamma, batch_size)


def train():
    for episode in range(episodes):

        state = env.reset()
        for step in range(steps):
            action = agent.greedy_action(state)
            next_state, reward, done, _ = env.step(action)

            reward = -1 if done else 0.1
            agent.store_transition(state, action, reward, next_state, done)
            agent.learn()
            state = next_state
            if done:
                break

        if episode % 100 == 0:
            total_reward = 0
            for i in range(test):
                state = env.reset()
                for j in range(steps):
                    env.render()
                    action = agent.action(state)
                    state, reward, done, _ = env.step(action)
                    total_reward += reward
                    if done:
                        break
            ave_reward = total_reward / test
            print('episode: ', episode, 'Evaluation Average Reward:', ave_reward)
    agent.save()


def evaluation():
    agent.restore()
    env.render()
    env.viewer.set_vsync(True)
    s = env.reset()
    while True:
        env.render()
        a = agent.action(s)
        s, r, done, _ = env.step(a)


if __name__ == '__main__':
    train()
    # evaluation()
