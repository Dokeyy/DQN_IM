"""
This part of code is the DQN brain, which is a brain of the agent.
All decisions are made in here.
Using Tensorflow to build the neural network.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.7.3
"""

import numpy as np
import pandas as pd
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            G_dict,
            q,
            #n_actions,
            #n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,

    ):
        #self.n_actions = n_actions
        #self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        self.G_dict = G_dict
        self.nodes_size = len(G_dict.keys())
        self.q = q # embedding维度

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, self.nodes_size * 2 + 2))

        # self.embedding_ph = np.zeros((self.nodes_size, self.q))
        self.embedding_ph = tf.zeros([self.nodes_size, self.q]) 

        self.sess = tf.Session()

        # consist of [target_net, evaluate_net]
        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        # self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.nodes_size], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.nodes_size], name='Q_target')  # for calculating loss

        # 节点embedding
        # self.embedding_ph = tf.placeholder(tf.float32, [self.nodes_size, self.q], name='embedding')

        with tf.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, \
                tf.random_normal_initializer(0.5, 0.3), tf.constant_initializer(0.1)  # config of layers

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                self.alpha1 = tf.get_variable('alpha1', [self.batch_size, 1], initializer=w_initializer, collections=c_names)
                self.alpha2 = tf.get_variable('alpha2', [self.batch_size, 1], initializer=w_initializer, collections=c_names)
                self.alpha3 = tf.get_variable('alpha3', [1, self.q], initializer=w_initializer, collections=c_names)
                self.alpha4 = tf.get_variable('alpha4', [self.batch_size, self.q], initializer=w_initializer, collections=c_names)
                # self.embedding_ph = tf.zeros([self.nodes_size, self.q])
                self.embedding_ph = self.cal_embedding(self.G_dict, self.s, self.embedding_ph, self.alpha1, self.alpha2, self.alpha3, self.alpha4, q=self.q)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                self.beta1 = tf.get_variable('beta1', [1, 2*self.q], initializer=w_initializer, collections=c_names)
                self.beta2 = tf.get_variable('beta2', [1, 1], initializer=w_initializer, collections=c_names)
                self.beta3 = tf.get_variable('beta3', [1, 1], initializer=w_initializer, collections=c_names)
                self.q_eval = self.cal_Q(self.G_dict, self.embedding_ph, self.beta1, self.beta2, self.beta3)

        # with tf.variable_scope('cal_embedding'):
        #    self.cal_embedding = cal_embedding(self.G, self.S, self.embedding_ph)

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)


        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.nodes_size], name='s_')    # input
        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                alpha1 = tf.get_variable('alpha1', [self.batch_size, 1], initializer=w_initializer, collections=c_names)
                alpha2 = tf.get_variable('alpha2', [self.batch_size, 1], initializer=w_initializer, collections=c_names)
                alpha3 = tf.get_variable('alpha3', [1, self.q], initializer=w_initializer, collections=c_names)
                alpha4 = tf.get_variable('alpha4', [self.batch_size, self.q], initializer=w_initializer, collections=c_names)
                
                self.embedding_ph = self.cal_embedding(self.G_dict, self.s, self.embedding_ph, alpha1, alpha2, alpha3, alpha4, q=self.q)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                beta1 = tf.get_variable('beta1', [1, 2*self.q], initializer=w_initializer, collections=c_names)
                beta2 = tf.get_variable('beta2', [1, 1], initializer=w_initializer, collections=c_names)
                beta3 = tf.get_variable('beta3', [1, 1], initializer=w_initializer, collections=c_names)
                self.q_next = self.cal_Q(self.G_dict, self.embedding_ph, beta1, beta2, beta3)

        # self.sess.run(tf.global_variables_initializer())

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        # newaxis在前面：[3] -> [1, 3]
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.nodes_size)
        return action

    def learn(self):
        # self.sess.run(tf.global_variables_initializer())
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=1)
        else:
            sample_index = np.random.choice(self.memory_counter, size=1)
        batch_memory = self.memory[sample_index, :]

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.nodes_size:],  # fixed params
                self.s: batch_memory[:, :self.nodes_size],  # newest params
            })

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.nodes_size].astype(int)
        reward = batch_memory[:, self.nodes_size + 1]

        #print('batch_index: ', batch_index)
        #print('q_next: ', q_next)
        #print('reward: ', reward.shape)
        #print('q_target: ', q_target)
        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next)

        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.nodes_size],
                                                self.q_target: q_target})

        
        self.cost_his.append(self.cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

    def cal_embedding(self, G_dict, S, embedding, alpha1, alpha2, alpha3, alpha4,  q=8, I=4, isEdgeWeighted=False):
        # 初始化q维0向量
        n = len(G_dict.keys());
        #print("n:", n)
        #print("S:", S)
        for i in range(I):
            # for si in range(S.shape[0]):
                # 计算每个节点
                for each_node in G_dict.keys():
                    print("cal_embedding,第{i}次,第{each_node}个节点".format(i=i, each_node=each_node))
                    # eyes = tf.transpose(tf.eye(self.nodes_size)[each_node])
                    # alpha4 = np.random.normal(1, 0.3, size=(q,1))
                    a_v = 0 if S[-1, each_node]==0 else 1
                    self_part = a_v*alpha4
                    if isEdgeWeighted==False:
                        # alpha2 = np.random.normal(1, 0.3, size=(1,1))
                        # alpha3 = np.random.normal(1, 0.3, size=(q,1))
                        weight_part = tf.matmul(alpha2, len(G_dict[each_node])*tf.nn.relu(alpha3))
                    # alpha1 = np.random.normal(1, 0.3, size=(1,1))
                    neighbors_sum = np.zeros((1, q))
                    for neighbors in G_dict[each_node]:
                        #print("neighbors_sum", neighbors_sum)
                        #print("embedding[neighbors]", embedding[neighbors])
                        #print(embedding[neighbors])
                        #print(neighbors_sum)
                        neighbors_sum += embedding[neighbors]
                    neighbor_part = tf.matmul(alpha1, neighbors_sum)
                    #print("neighbor_part", neighbor_part.shape)
                    #print("weight_part", tf.Session().run(weight_part).T.shape)
                    #print("self_part", self_part.T.shape)
                    part1 = embedding[:each_node]
                    part2 = embedding[each_node+1:]
                    embedding = tf.concat([part1, neighbor_part + weight_part + self_part, part2], axis=0)
                    # embedding[each_node] = tf.nn.relu(neighbor_part + weight_part + self_part)
                #print("已迭代{i}次".format(i=i))
                #print("embedding", sum(embedding))
        return embedding

    def cal_Q(self, G_dict, embedding, beta1, beta2, beta3):
        Q = tf.zeros([self.batch_size, self.nodes_size])
        for each_node in G_dict.keys():
            print("cal_Q,第{each_node}个节点".format(each_node=each_node))
            self_part = tf.tile(embedding[each_node] * beta3, [self.batch_size, 1])
            neightbor_sum = np.zeros((self.batch_size, self.q))
            for neighbors in G_dict[each_node]:
                neightbor_sum = neightbor_sum + embedding[neighbors]
            neighbor_part = beta2 * neightbor_sum
            part1 = Q[:, :each_node]
            #print('part1 ', part1)
            part2 = Q[:, each_node+1:]
            #print('part2 ', part2)
            Q = tf.concat([part1, tf.transpose(tf.matmul(beta1, tf.transpose(tf.nn.relu(tf.concat([neighbor_part, self_part], axis=1))))), part2], axis=1)
        return Q
