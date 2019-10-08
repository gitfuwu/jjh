import tensorflow as tf
import numpy as np
import random
from collections import deque


class DuelingDQN:
    def __init__(self, env, lr, ini_epsilon, decay_steps, replay_size, gamma, batch_size, update_frequency):
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.lr = lr
        self.ini_epsilon = ini_epsilon
        self.decay_steps = decay_steps
        self.replay_size = replay_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.update_frequency = update_frequency

        self.replay_buffer = deque()
        self.train_num = 0

        self.state_input = tf.placeholder(tf.float32, [None, self.state_dim])
        self.y_input = tf.placeholder(tf.float32, [None])
        self.action_input = tf.placeholder(tf.float32, [None, self.action_dim])

        self.build_network()

        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())
        self.session.run(self.target_replace_op)

    def store_transition(self, state, action, reward, next_state, done):
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        if len(self.replay_buffer) >= self.replay_size:
            self.replay_buffer.popleft()
        self.replay_buffer.append((state, one_hot_action, reward, next_state, done))

    def learn(self):
        if len(self.replay_buffer) > self.batch_size:
            self.train_num += 1
            mini_batch = random.sample(self.replay_buffer, self.batch_size)
            state_batch = [data[0] for data in mini_batch]
            action_batch = [data[1] for data in mini_batch]
            reward_batch = [data[2] for data in mini_batch]
            next_state_batch = [data[3] for data in mini_batch]

            y_batch = []
            q_value_batch = self.target_q_value.eval(feed_dict={self.state_input: next_state_batch})
            current_q_value_batch = self.q_value.eval(feed_dict={self.state_input: next_state_batch})
            max_action_next = np.argmax(current_q_value_batch, axis=1)
            for i in range(self.batch_size):
                done = mini_batch[i][4]
                if done:
                    y_batch.append(reward_batch[i])
                else:
                    y_batch.append(reward_batch[i] + self.gamma * q_value_batch[i, max_action_next[i]])

            self.optimizer.run(feed_dict={
                self.y_input: y_batch,
                self.action_input: action_batch,
                self.state_input: state_batch
            })
            if self.train_num % self.update_frequency == 0:
                self.session.run(self.target_replace_op)

    def greedy_action(self, state):
        q_value = self.q_value.eval(feed_dict={
            self.state_input: [state]
        })[0]
        epsilon = self.ini_epsilon - self.ini_epsilon / self.decay_steps * self.train_num
        if random.random() <= epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            return np.argmax(q_value)

    def action(self, state):
        return np.argmax(self.q_value.eval(feed_dict={
            self.state_input: [state]
        })[0])

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

    def save(self):
        saver = tf.train.Saver()
        saver.save(self.session, './model/params', write_meta_graph=False)

    def restore(self):
        saver = tf.train.Saver()
        saver.restore(self.session, './model/params')

    def build_network(self):
        with tf.variable_scope('current_net'):
            w1 = self.weight_variable([self.state_dim, 20])
            b1 = self.bias_variable([20])
            w_alpha = self.weight_variable([20, 1])
            b_alpha = self.bias_variable([1])
            w_beta = self.weight_variable([20, self.action_dim])
            b_beta = self.weight_variable([self.action_dim])

            h_layer = tf.nn.relu(tf.matmul(self.state_input, w1) + b1)
            self.current_value = tf.matmul(h_layer, w_alpha) + b_alpha
            self.current_advantage = tf.matmul(h_layer, w_beta) + b_beta

            self.q_value = self.current_value + (
                    self.current_advantage - tf.reduce_mean(self.current_advantage, axis=1, keep_dims=True))
            self.q_action = tf.reduce_sum(tf.multiply(self.q_value, self.action_input), reduction_indices=1)
            self.loss = tf.reduce_mean(tf.square(self.y_input - self.q_action))
            self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        with tf.variable_scope('target_net'):
            w1 = self.weight_variable([self.state_dim, 20])
            b1 = self.bias_variable([20])
            w_alpha = self.weight_variable([20, 1])
            b_alpha = self.bias_variable([1])
            w_beta = self.weight_variable([20, self.action_dim])
            b_beta = self.weight_variable([self.action_dim])

            h_layer = tf.nn.relu(tf.matmul(self.state_input, w1) + b1)

            self.target_value = tf.matmul(h_layer, w_alpha) + b_alpha
            self.target_advantage = tf.matmul(h_layer, w_beta) + b_beta

            self.target_q_value = self.target_value + (
                    self.target_advantage - tf.reduce_mean(self.target_advantage, axis=1, keep_dims=True))

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='current_net')

        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
