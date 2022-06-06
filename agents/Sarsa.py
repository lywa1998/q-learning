import torch
import numpy as np


class Sarsa:
    def __init__(self,
                 obs_dim,
                 action_dim,
                 learning_rate=0.5,
                 gamma=1,
                 epsilon=0.1):
        self.action_dim = action_dim  # 动作维度，有几个动作可选
        self.lr = learning_rate  # 学习率
        self.gamma = gamma  # reward 的衰减率
        self.epsilon = epsilon  # 按一定概率随机选动作，即 e-greedy 策略
        self.Q_table = np.zeros((obs_dim, action_dim))  # 初始化Q表格

    # 根据输入观察值，采样输出的动作值，带探索(epsilon-greedy，训练时用这个方法)
    def sample(self, obs):
        if np.random.uniform(0, 1) > self.epsilon:  #根据table的Q值选动作
            action = self.predict(obs)
        else:
            action = np.random.choice(self.action_dim)  #有一定概率随机探索选取一个动作
        return action

    # 根据输入观察值，预测输出的动作值（已有里面挑最大，贪心的算法，只有利用，没有探索）
    def predict(self, obs):
        Q_list = self.Q_table[obs, :]
        maxQ = np.max(Q_list)  # 找到最大Q对应的下标
        action_list = np.where(Q_list == maxQ)[0]  # maxQ可能对应多个action
        action = np.random.choice(action_list)  # 从这些action中随机挑一个action（可以打印出来看看）
        return action

    # 学习方法，也就是更新Q-table的方法
    def learn(self, obs, action, reward, next_obs, next_action, done):
        """ on-policy
            obs: 交互前的obs, s_t
            action: 本次交互选择的action, a_t
            reward: 本次动作获得的奖励r
            next_obs: 本次交互后的obs, s_t+1
            next_action: 根据当前Q表格, 针对next_obs会选择的动作, a_t+1
            done: episode是否结束
        """
        predict_Q = self.Q_table[obs, action]
        if done:  # done为ture的话，代表这是episode最后一个状态
            target_Q = reward  # 没有下一个状态了
        else:
            target_Q = reward + self.gamma * self.Q_table[next_obs, next_action]  # Sarsa
        self.Q_table[obs, action] += self.lr * (target_Q - predict_Q)  # 修正q

    def save_model(self,path):
        '''把 Q表格 的数据保存到文件中
        '''
        np.save(path, self.Q_table)

    def load_model(self, path):
        '''从文件中读取数据到 Q表格
        '''
        self.Q_table = np.load(path)
