import tensorflow as tf
import numpy as np
import random


class SumTree(object):
    """
    This SumTree code is a modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py
    Story data with its priority in the tree.
    """
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame

        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:  # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:  # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1  # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):  # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:  # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This Memory class is modified based on the original code from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.leaf_node_num = 0

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)  # set the max p for new p
        self.leaf_node_num += 1

    def sample(self, n):
        b_idx, b_memory, weights = np.empty((n,), dtype=np.int32), np.empty((n, self.tree.data[0].size)), np.empty(
            (n, 1))
        pri_seg = self.tree.total_p / n  # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p  # for later calculate ISweight
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            weights[i, 0] = np.power(prob / min_prob, -self.beta)
            b_idx[i], b_memory[i, :] = idx, data
        return b_idx, b_memory, weights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)


class PDQN:
    def __init__(self, env, lr, ini_epsilon, decay_steps, replay_size, gamma, batch_size, update_frequency):
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.lr = lr
        self.ini_epsilon = ini_epsilon
        self.decay_steps = decay_steps
        self.gamma = gamma
        self.batch_size = batch_size
        self.update_frequency = update_frequency
        self.replay_size = replay_size

        self.memory = Memory(capacity=self.replay_size)
        self.train_num = 0

        self.state_input = tf.placeholder(tf.float32, [None, self.state_dim])
        self.y_input = tf.placeholder(tf.float32, [None, 1])
        self.action_input = tf.placeholder(tf.float32, [None, self.action_dim])
        self.weights = tf.placeholder(tf.float32, [None, 1])

        self.build_network()

        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())
        self.session.run(self.target_replace_op)

    def store_transition(self, state, action, reward, next_state, done):
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        transition = np.hstack((state, one_hot_action, reward, next_state, done))
        self.memory.store(transition)

    def learn(self):
        if self.memory.leaf_node_num > self.replay_size:
            self.train_num += 1
            tree_idx, mini_batch, weights = self.memory.sample(self.batch_size)
            state_batch = mini_batch[:, 0:self.state_dim]
            action_batch = mini_batch[:, self.state_dim:self.state_dim + self.action_dim]
            reward_batch = mini_batch[:, self.state_dim + self.state_dim + self.action_dim]
            next_state_batch = mini_batch[:, -self.state_dim:]

            y_batch = np.zeros((self.batch_size, 1))
            q_value_batch = self.target_q_value.eval(feed_dict={self.state_input: next_state_batch})
            current_q_value_batch = self.q_value.eval(feed_dict={self.state_input: next_state_batch})
            max_action_next = np.argmax(current_q_value_batch, axis=1)
            for i in range(self.batch_size):
                done = mini_batch[i][4]
                if done:
                    y_batch[i][0] = reward_batch[i]
                else:
                    y_batch[i][0] = reward_batch[i] + self.gamma * q_value_batch[i, max_action_next[i]]

            self.optimizer.run(feed_dict={
                self.y_input: y_batch,
                self.action_input: action_batch,
                self.state_input: state_batch,
                self.weights: weights
            })

            abs_errors = self.session.run(self.abs_errors, feed_dict={
                self.y_input: y_batch,
                self.action_input: action_batch,
                self.state_input: state_batch
            })
            self.memory.batch_update(tree_idx, abs_errors)  # update priority

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
            w2 = self.weight_variable([20, self.action_dim])
            b2 = self.bias_variable([self.action_dim])

            h_layer = tf.nn.relu(tf.matmul(self.state_input, w1) + b1)
            self.q_value = tf.matmul(h_layer, w2) + b2
            self.q_action = tf.reduce_sum(tf.multiply(self.q_value, self.action_input), reduction_indices=1,
                                          keep_dims=True)
            self.loss = tf.reduce_mean(self.weights * tf.square(self.y_input - self.q_action))
            self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
            self.abs_errors = tf.abs(self.y_input - self.q_action)

        with tf.variable_scope('target_net'):
            w1 = self.weight_variable([self.state_dim, 20])
            b1 = self.bias_variable([20])
            w2 = self.weight_variable([20, self.action_dim])
            b2 = self.bias_variable([self.action_dim])

            h_layer = tf.nn.relu(tf.matmul(self.state_input, w1) + b1)
            self.target_q_value = tf.matmul(h_layer, w2) + b2

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='current_net')

        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
