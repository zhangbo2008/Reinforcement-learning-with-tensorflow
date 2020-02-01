"""
A simple example for Reinforcement Learning using table lookup Q-learning method.
An agent "o" is on the left of a 1 dimensional world, the treasure is on the rightmost location.
Run this program and to see how the agent will improve its strategy of finding the treasure.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

从莫凡python开始学习rl.2020-02-01,14点05
有代码有视频,有公式.必须靠谱的教程.把学习过程直接写到这套代码的注释里面.

"""

import numpy as np
import pandas as pd
import time

np.random.seed(2)  # reproducible


N_STATES = 6   # the length of the 1 dimensional world
ACTIONS = ['left', 'right']     # available actions
EPSILON = 0.9   # greedy police
ALPHA = 0.1     # learning rate
GAMMA = 0.9    # discount factor
MAX_EPISODES = 13   # maximum episodes
FRESH_TIME = 0.3    # fresh time for one move


def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),     # q_table initial values 初始化全是0
        #因为默认是没有value的.没有奖励和惩罚.
        columns=actions,    # actions's name
    )
    # print(table)    # show table
    return table


def choose_action(state, q_table):
    # This is how to choose an action
    state_actions = q_table.iloc[state, :]   #当前状态的action表.也就是q表的其中一行.
    if (np.random.uniform() > EPSILON) or ((state_actions == 0).all()):  # act non-greedy or state-action have no value
        action_name = np.random.choice(ACTIONS)
    else:   # act greedy
        action_name = state_actions.idxmax()    # replace argmax to idxmax as argmax means a different function in newer version of pandas
    return action_name


def get_env_feedback(S, A):
    # This is how agent will interact with the environment
    if A == 'right':    # move right
        if S == N_STATES - 2:   # terminate     0 ,1,2,3,4,5
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1
            R = 0
    else:   # move left
        R = 0
        if S == 0:
            S_ = S  # reach the wall
        else:
            S_ = S - 1
    return S_, R


def update_env(S, episode, step_counter):
    # This is how environment be updated
    env_list = ['-']*(N_STATES-1) + ['T']   # '---------T' our environment
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)





#核心就是rl函数!!!!!!!!!!!!!!!!!!!也是程序的入口
def rl():
    # main part of RL loop  建立初始化环境.
    q_table = build_q_table(N_STATES, ACTIONS)



    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = 0
        is_terminated = False
        update_env(S, episode, step_counter) #利用这个update_env函数来画出来s的当前状态. 也就是可视化函数.
        while not is_terminated:  # 到达目标就停止这次迭代

            A = choose_action(S, q_table) #根据 一部禧龙-贪婪的方法根据q表选一个action.


            #根据s和a计算出s',这里面就是s_   ,表示a作用后得到的新s 和对应的r ,r 这个函数是需要人工
            #制定的. 这里面s',到terminal 是1, 其他都是0.
            S_, R = get_env_feedback(S, A)  # take action & get next state and reward



            q_predict = q_table.loc[S, A] # q预测就是当前q表给的数值.


            #下面计算q真实: q_真实=当前r+ 未来衰减系数* 未来收获的max


            # 为什么这么算是对的呢??

            '''
            
              q_target = R + GAMMA * q_table.iloc[S_, :].max() 
              核心就是这行代码.
              表示了q_真实应该这么算.为什么呢?
              这行本质是按照Gamma概率选择q表中S_对应的action中Q值最大的action,1-Gamma概率选择其他action.
              所以是局部贪心.
              
              证明的核心是要证明,是否能让最后一步得到的1.体现在之前的每一步中.
              下面取特殊的gamma=1为了计算方便.不影响结论的一般性.
              1.首先证明特苏情况,整个直线只有2步,一步开始,一步terminal.所以开局,right能学到1.
              2.然后归纳法继续看.如果整个游戏对的直线有3步(state:记作a,b,c) c是terminal
                  那么一开始怎么传都是0,因为不可能一步转移就到terminal.传一次之后:
                  q表:
                     left right 
                  a   0     0
                  b   0     0
                  c   0     0
                  然后b再传一次,
                  q表:
                     left right 
                  a   0     0
                  b   0     1
                  c   0     0
                  这样第一次循环就结束了.
                  第二次循环.
                  q表:
                     left right 
                  a   0     1
                  b   0     1
                  c   0     0
                  从而看出来,按照循环,确实从R和 q_table.iloc[S_, :].max() 
                  把权重参数按照逆流程传送到了开始的步骤,所以算法有效.证毕!!!!!!!!!!!!!!!!!!
            ps:虽然是根据一个例子来做的q learning的证明,但是证明过程完全可以抽象画,不失一般性.
            从而这种q_learning算法对于任何强化问题都适用.证毕.
              
            
              
              
              
            q_predict = q_table.loc[S, A]
            
            
            '''







            if S_ != 'terminal':
                q_target = R + GAMMA * q_table.iloc[S_, :].max()   # next state is not terminal
            else:
                q_target = R     # next state is terminal
                is_terminated = True    # terminate this episode

            q_table.loc[S, A] += ALPHA * (q_target - q_predict)  # update
            S = S_  # move to next state

            update_env(S, episode, step_counter+1)
            step_counter += 1
    return q_table


if __name__ == "__main__":
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)

    '''
           left     right
0  0.000000  0.004320
1  0.000000  0.025005
2  0.000030  0.111241
3  0.000000  0.368750
4  0.027621  0.745813
5  0.000000  0.000000
这个是最后打印的q表.跟想想的一样.都是right的权重更高.并且成衰减泰式.因为gamma的值是小于1的
经过实验知道,gamma取的越小,衰减的越厉害.
    '''
