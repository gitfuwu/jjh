from AC import Actor, Critic
import tensorflow as tf
import gym

ENV_NAME = 'CartPole-v0'
EPISODE = 3000  # Episode limitation
STEP = 3000  # Step limitation in an episode
TEST = 10  # The number of experiment test every 100 episode

critic_lr = 0.01
critic_gamma = 0.9

actor_lr = 0.01


def main():
    # initialize OpenAI Gym env and dqn agent
    sess = tf.InteractiveSession()
    env = gym.make(ENV_NAME)
    actor = Actor(env, sess, actor_lr)
    critic = Critic(env, sess, critic_lr, critic_gamma)

    for episode in range(EPISODE):
        state = env.reset()
        # Train
        for step in range(STEP):
            action = actor.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            td_error = critic.learn(state, reward, next_state)
            actor.learn(state, action, td_error)
            state = next_state
            if done:
                break

        # Test every 100 episodes
        if episode % 100 == 0:
            total_reward = 0
            for i in range(TEST):
                state = env.reset()
                for j in range(STEP):
                    env.render()
                    action = actor.choose_action(state)  # direct action for test
                    state, reward, done, _ = env.step(action)
                    total_reward += reward
                    if done:
                        break
            ave_reward = total_reward / TEST
            print('episode: ', episode, 'Evaluation Average Reward:', ave_reward)


if __name__ == '__main__':
    main()
