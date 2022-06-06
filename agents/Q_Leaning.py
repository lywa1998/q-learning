import torch
import numpy as np
import random


class QLearning:
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

    def sample(self, obs):
        '''根据输入观测值，采样输出的动作值，带探索，训练模型时使用
        '''
        if random.random() >= self.epsilon:  # 随机选取0-1之间的值，如果大于等于epsilon就按照贪心策略选取action，否则随机选取
            action = self.predict(obs)
        else:
            action = np.random.choice(self.action_dim)  #有一定概率随机探索选取一个动作
        return action

    def predict(self, obs):
        '''根据输入观测值，采样输出的动作值，不带探索，测试模型时使用
        '''
        Q_list = self.Q_table[obs, :]
        Q_max = np.max(Q_list)
        action_list = np.where(Q_list == Q_max)[0]
        action = np.random.choice(action_list) # Q_max可能对应多个 action ，可以随机抽取一个
        return action

    def learn(self, obs, action, reward, next_obs, done):
        '''学习方法(off-policy)，也就是更新Q-table的方法
        Args:
            obs [type]: 交互前的obs, s_t
            action [type]: 本次交互选择的action, a_t
            reward [type]: 本次动作获得的奖励r
            next_obs [type]: 本次交互后的obs, s_t+1
            done function:  episode是否结束
        '''
        Q_predict = self.Q_table[obs, action]
        if done:
            Q_target = reward  # 没有下一个状态了
        else:
            Q_target = reward + self.gamma * np.max(self.Q_table[next_obs, :])  # Q_table-learning
        self.Q_table[obs, action] += self.lr * (Q_target - Q_predict)  # 修正q

    def save_model(self,path):
        '''把 Q表格 的数据保存到文件中
        '''
        np.save(path, self.Q_table)

    def load_model(self, path):
        '''从文件中读取数据到 Q表格
        '''
        self.Q_table = np.load(path)