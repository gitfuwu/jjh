import gym
from DoubleDQN import DoubleDQN

env_name = 'CartPole-v0'
episodes = 3000
steps = 300
test = 10

lr = 0.0001
ini_epsilon = 0.3
decay_steps = 200000  # 调用agent中learn方法decay_steps后，ini_epsilon会衰减到0
replay_size = 100
gamma = 0.9
batch_size = 32
update_frequency = 1
env = gym.make(env_name)
agent = DoubleDQN(env, lr, ini_epsilon, decay_steps, replay_size, gamma, batch_size, update_frequency)


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
    s = env.reset()
    total_reward = 0
    while True:
        env.render()
        a = agent.action(s)
        s, r, done, _ = env.step(a)
        total_reward += r
        if done:
            break
    print("total reward ", total_reward)


if __name__ == '__main__':
    train()
    # evaluation()
