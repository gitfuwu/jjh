import tensorflow as tf
import numpy as np


class Actor:
    def __init__(self, env, sess, lr):
        # init some parameters
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.lr = lr

        self.state_input = tf.placeholder(tf.float32, [None, self.state_dim])
        self.action_input = tf.placeholder(tf.float32, [None, self.action_dim])
        self.td_error = tf.placeholder(tf.float32, [None, 1])

        self.build_network()
        # Init session
        self.session = sess
        self.session.run(tf.global_variables_initializer())

    def build_network(self):
        w1 = self.weight_variable([self.state_dim, 20])
        b1 = self.bias_variable([20])
        w2 = self.weight_variable([20, self.action_dim])
        b2 = self.bias_variable([self.action_dim])

        h_layer = tf.nn.relu(tf.matmul(self.state_input, w1) + b1)
        softmax_input = tf.matmul(h_layer, w2) + b2

        self.all_act_prob = tf.nn.softmax(softmax_input)
        self.neg_log_prob = -tf.log(
            tf.reduce_sum(tf.multiply(self.all_act_prob, self.action_input), reduction_indices=1, keep_dims=True))
        self.loss = tf.reduce_mean(self.neg_log_prob * self.td_error)
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(-self.loss)

    @staticmethod
    def weight_variable(shape):
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial)

    @staticmethod
    def bias_variable(shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

    def choose_action(self, observation):
        prob_weights = self.session.run(self.all_act_prob, feed_dict={self.state_input: observation[np.newaxis, :]})
        action = np.random.choice(range(prob_weights.shape[1]),
                                  p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action

    def learn(self, state, action, td_error):
        s = state[np.newaxis, :]
        one_hot_action = np.zeros((1, self.action_dim))
        one_hot_action[0, action] = 1
        # train on episode
        self.session.run(self.train_op, feed_dict={
            self.state_input: s,
            self.action_input: one_hot_action,
            self.td_error: td_error
        })


class Critic:
    def __init__(self, env, sess, lr, gamma):
        # init some parameters
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.lr = lr
        self.gamma = gamma

        self.state_input = tf.placeholder(tf.float32, [1, self.state_dim])
        self.value_ = tf.placeholder(tf.float32, [1, 1])
        self.reward = tf.placeholder(tf.float32, [1, 1])

        self.build_network()

        # Init session
        self.session = sess
        self.session.run(tf.global_variables_initializer())

    def build_network(self):
        # network weights
        w1 = self.weight_variable([self.state_dim, 20])
        b1 = self.bias_variable([20])
        w2 = self.weight_variable([20, 1])
        b2 = self.bias_variable([1])
        h_layer = tf.nn.relu(tf.matmul(self.state_input, w1) + b1)

        self.value = tf.matmul(h_layer, w2) + b2
        self.td_error = self.reward + self.gamma * self.value_ - self.value
        self.loss = tf.square(self.td_error)
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def learn(self, state, reward, next_state):
        reward = np.reshape(reward, [1, 1])
        s, s_ = state[np.newaxis, :], next_state[np.newaxis, :]
        v_ = self.session.run(self.value, {self.state_input: s_})
        td_error, _ = self.session.run([self.td_error, self.train_op],
                                       {self.state_input: s, self.value_: v_, self.reward: reward})
        return td_error

    @staticmethod
    def weight_variable(shape):
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial)

    @staticmethod
    def bias_variable(shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)
