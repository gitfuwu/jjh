from PGMC import PGMC
import gym

ENV_NAME = 'CartPole-v0'
EPISODE = 3000  # Episode limitation
STEP = 3000  # Step limitation in an episode
TEST = 10  # The number of experiment test every 100 episode

lr = 0.01
gamma = 0.9


def main():
    env = gym.make(ENV_NAME)
    agent = PGMC(env, lr, gamma)

    for episode in range(EPISODE):
        state = env.reset()
        for step in range(STEP):
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward)
            state = next_state
            if done:
                agent.learn()
                break

        # Test every 100 episodes
        if episode % 100 == 0:
            total_reward = 0
            for i in range(TEST):
                state = env.reset()
                for j in range(STEP):
                    env.render()
                    action = agent.choose_action(state)  # direct action for test
                    state, reward, done, _ = env.step(action)
                    total_reward += reward
                    if done:
                        break
            ave_reward = total_reward / TEST
            print('episode: ', episode, 'Evaluation Average Reward:', ave_reward)


if __name__ == '__main__':
    main()
