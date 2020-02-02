"""
This part of code is the Q learning brain, which is a brain of the agent.
All decisions are made in here.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

import numpy as np
import pandas as pd


class RL(object):
    def __init__(self, action_space, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = action_space  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy

        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

    def choose_action(self, observation):
        self.check_state_exist(observation)
        # action selection
        if np.random.rand() < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[observation, :]
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    def learn(self, *args):
        pass


# off-policy
class QLearningTable(RL):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super(QLearningTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update


# on-policy
class SarsaTable(RL):

    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super(SarsaTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

    def learn(self, s, a, r, s_, a_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, a_]  # next state is not terminal

            # 跟q_leran区别就是  self.q_table.loc[s_, a_]  这一点.
        #     效果好像收敛比q_learn要慢.因为他每一步走的不是max,而是百分之90走max.就是本py的39行.
            '''
            所以,sarsa 走 当前局部最优路径概率是90,q学习走当前局部最优路径是百分百.
            所以sarsa走到-1开始的概率更高,会在开始学习的时候到达最优解的速度更慢.
            但是他到达-1后就会学习到这个负的权重,从而以后会更大概率的避免掉入陷阱中.
            
            下面还是证明sarsa算法
            
            
            证明:
            假设图中只有3个坐标点,然后a,b,c开始在b点.到a就死,到c就活.
            那么sarsa一开始随机走一步假设到a了.然后sarsa表就更新b,left=-1
            所以可以达到学习效果
            下面假设图中有5个坐标点a,b,c,d,e 开始在c点.到a就是死.到e就是活
            那么根据sarsa.第一次走到a就得到b,left就是-1 游戏结束
            然后第二次走到b,如果又走了left就会把b,left的信息更新到c,left 里面.
            这点还是从公式上看出来.就是下面公式的最后一项.
            q_target = r + self.gamma * self.q_table.loc[s_, a_]
            
            然后通过下面一行代码把这些东西加到c,left里面.
            self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update
            所以证毕.sarssa也是会学习到最后的最优策略.
            
            整体证明过程跟q_learn很相似.
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            '''






        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update
