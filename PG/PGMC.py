import tensorflow as tf
import numpy as np


class PGMC:
    def __init__(self, env, lr, gamma):
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.lr = lr
        self.gamma = gamma

        self.ep_s_list = list()
        self.ep_a_list = list()
        self.ep_r_list = list()

        # network input
        self.state_input = tf.placeholder(tf.float32, [None, self.state_dim])
        self.action_input = tf.placeholder(tf.float32, [None, self.action_dim])
        self.v_input = tf.placeholder(tf.float32, [None, 1])

        self.build_network()

        self.session = tf.InteractiveSession()
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
        self.loss = tf.reduce_mean(self.neg_log_prob * self.v_input)
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def store_transition(self, s, a, r):
        self.ep_s_list.append(s)
        one_hot_a = np.zeros((1, self.action_dim))
        one_hot_a[0, a] = 1
        self.ep_a_list.append(one_hot_a)
        self.ep_r_list.append(r)

    def choose_action(self, s):
        prob_weights = self.session.run(self.all_act_prob, feed_dict={self.state_input: s[np.newaxis, :]})
        action = np.random.choice(range(prob_weights.shape[1]),
                                  p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action

    def learn(self):
        discounted_ep_r_list = np.zeros((len(self.ep_r_list), 1))
        running_add = 0
        for t in reversed(range(len(self.ep_r_list))):
            running_add = running_add * self.gamma + self.ep_r_list[t]
            discounted_ep_r_list[t][0] = running_add
        discounted_ep_r_list -= np.mean(discounted_ep_r_list)
        discounted_ep_r_list /= np.std(discounted_ep_r_list)

        self.session.run(self.train_op, feed_dict={
            self.state_input: np.vstack(self.ep_s_list),
            self.action_input: np.vstack(self.ep_a_list),
            self.v_input: discounted_ep_r_list
        })

        self.ep_s_list, self.ep_a_list, self.ep_r_list = [], [], []

    @staticmethod
    def weight_variable(shape):
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial)

    @staticmethod
    def bias_variable(shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

    def save(self):
        saver = tf.train.Saver()
        saver.save(self.session, './model/params', write_meta_graph=False)

    def restore(self):
        saver = tf.train.Saver()
        saver.restore(self.session, './model/params')
