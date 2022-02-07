import numpy as np
import pandas as pd
import tensorflow as tf
from RL_brain_bishe import DeepQNetwork
import networkx as nx
from nodes_env import Env
import random

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def load_net(path):
    G = nx.read_edgelist(path, nodetype=int)
    print("Loading Success!")
    return G

def relu(inX):
    return np.maximum(0,inX)

def cal_embedding(G_dict, alpha1, alpha2, alpha3, alpha4, q=32, I=4, isEdgeWeighted=False):
    # 初始化q维0向量
    n = len(G_dict.keys())
    #print("n:", n)
    embedding = np.zeros((n, q))

    # 初始化S种子集合
    S = [0] * n

    # 迭代I次计算节点向量
    # 迭代I次
    for i in range(I):
        # 计算每个节点
        for each_node in G_dict.keys():
            # alpha4 = np.random.normal(1, 0.3, size=(q,1))
            if(S[each_node]==1):
                a_v = 1
            else:
                a_v = 0
            self_part = a_v*alpha4
            if isEdgeWeighted==False:
                # alpha2 = np.random.normal(1, 0.3, size=(1,1))
                # alpha3 = np.random.normal(1, 0.3, size=(q,1))
                weight_part = alpha2*len(G_dict[each_node])*relu(alpha3)
            # alpha1 = np.random.normal(1, 0.3, size=(1,1))
            neighbors_sum = np.zeros((1, q))
            for neighbors in G_dict[each_node]:
                neighbors_sum += embedding[neighbors]
            #print(neighbors_sum.shape)
            neighbor_part = np.array([alpha1*data for data in neighbors_sum])
            #print(neighbor_part.shape)
            #print(weight_part.shape)
            #print(self_part.shape)
            embedding[each_node] = relu((neighbor_part + weight_part)+ self_part.T)
        print("已迭代{i}次，embedding结果为：{embedding}".format(i=i, embedding=embedding))
    return embedding

def cal_Q(G_dict, q, embedding, beta1, beta2, beta3):
    n = len(G_dict.keys())
    Q = np.zeros(n)
    for each_node in G_dict.keys():
        self_part = np.array([data * beta3[0][0] for data in embedding[each_node]])
        neighbor_sum = np.zeros((1, q))
        for neighbors in G_dict[each_node]:
            neighbor_sum = neighbor_sum + embedding[neighbors]
            # print(neighbor_sum.shape)
            # print(beta2)
        neighbor_part = np.array([data * beta2[0][0] for data in neighbor_sum])
        # print(neighbor_part.shape)
        # print(self_part.shape)
        Q[each_node] = beta1.dot(relu(np.concatenate((self_part.reshape((1, q)), neighbor_part),axis=1)).T)
    return Q


if __name__ == "__main__":
    G = load_net("/Users/dssa/Downloads/test_littlenet.txt")
    #cal_embedding(G.subgraph(dict(nx.bfs_successors(G, 1, 2)).keys()), N=G.number_of_nodes())
    # 将G转化为字典形式
    G_dict = {}
    for each_node in G.nodes():
        for neighbors in G.neighbors(each_node):
            if each_node in G_dict:
                G_dict[each_node].append(neighbors)
            else:
                G_dict[each_node] = [neighbors]

    env = Env()
    Dqn = DeepQNetwork(G_dict, 8)

    step = 0
    for i in range(1000):
        print('---------第{i}次---------'.format(i=i))
        observation = env.reset(G.number_of_nodes())
        for j in range(10):
            action = Dqn.choose_action(observation)
            observation_, reward = env.step(action)
            Dqn.store_transition(observation, action, reward, observation_)
            
            if (step > 200) and (step % 5 == 0):
                Dqn.learn()
            
            observation = observation_

            step += 1

    print("alpha1: ", Dqn.sess.run(Dqn.alpha1))
    # Dqn.plot_cost()
    print('Training Over.')

    # 保存权重csv
    rand = random.randint(0, Dqn.batch_size)
    print("rand: ", rand)
    alpha1 = Dqn.sess.run(Dqn.alpha1)
    alpha2 = Dqn.sess.run(Dqn.alpha2)
    alpha3 = Dqn.sess.run(Dqn.alpha3)
    alpha4 = Dqn.sess.run(Dqn.alpha4)
    beta1 = Dqn.sess.run(Dqn.beta1)
    beta2 = Dqn.sess.run(Dqn.beta2)
    beta3 = Dqn.sess.run(Dqn.beta3)
    np.savetxt("alpha1.csv",  alpha1, delimiter=',')
    np.savetxt("alpha2.csv",  alpha2, delimiter=',')
    np.savetxt("alpha3.csv",  alpha3, delimiter=',')
    np.savetxt("alpha4.csv",  alpha4, delimiter=',')
    np.savetxt("beta1.csv",  beta1, delimiter=',')
    np.savetxt("beta2.csv",  beta2, delimiter=',')
    np.savetxt("beta3.csv",  beta3, delimiter=',')
    embedding = cal_embedding(G_dict, alpha1[rand], alpha2[rand], alpha3, alpha4[rand], q=8)
    print(embedding)
    Q_value = cal_Q(G_dict, 8, embedding, beta1, beta2, beta3)
    print(Q_value)
    print(np.abs(Q_value))



