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
            n_actions,           #4
            n_features,           #2
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # consist of [target_net, evaluate_net]
        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')

        #下行,把e赋值给t跟  把eval 的数值都赋值给target
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):
        # ------------------ build evaluate_net ------------------


        #输入是特征. eval_net生成q表.输入特征,输出一组q值,对应于action数量.

        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        with tf.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            # first layer. collections is used later when assign to target net

            # 全链接.
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l1, w2) + b2

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
















        # 核心是q_target 怎么来的.?
        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):#初始化.
            self.memory_counter = 0
#memory 中存的数据是:s, [a, r], s_  进行水平拼接.所以结果是 s[0],s[1],a,r,s_[0],s[1]
        transition = np.hstack((s, [a, r], s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        # 首先补一下batch_size这一个维度.下面就是补出一个新维度.当batch_size,补成一样的才能放入网络中传到.
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions

            # 所以这个网络q_eval就表示之前q学习里面的q表中对应输出各个action对应的q值的一行.
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            #action就是直接取最大.


            # 对应第二个项目2_Q_learning_maze进行比较学习.就更直白了.
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action




#下面这个learn就是核心了.qdn算法.
    def learn(self):


        #首先看当前learn步奏是否需要参数赋值.从e赋值到target里面.
        # check to replace target parameters


        # 也就是说q_target 里面的权重更新的频率很慢,
        #  每self.replace_target_iter 多次才更新一遍.
        '''
        为什么要这么慢速的更新呢?

loss是 最后是traget





        :return:
        '''





        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')



#下面这段,从记忆体中取出batch_size这些sample对象.
        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            #当目前迭代的超过size了,就在size里面choice
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            #当前如果没超过size,就子啊counter里面choice
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)

        batch_memory = self.memory[sample_index, :]








#从s 计算出 q_eval   s_ 计算出q_next   他俩网络一样但是参数不一样.

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:],  # fixed params
                self.s: batch_memory[:, :self.n_features],  # newest params
            })

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)   #对应的动作 是一个整数.
        reward = batch_memory[:, self.n_features + 1]  #对应的奖励.




# 类比q学习:q_target = r + self.gamma * self.q_table.loc[s_, :].max()
        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        '''
        
        从这一系列看,q表怎么获得,完全随意,只是从特征,然后函数拟合器出来一个实数就行.
        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)  公式*
        这行代码是关键.
        这行代码会吧生成出来的q值,根据reward进行修正.
        
        需呀证明这么算是对的..
        
        
         self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
         
         
         loss是算target和eval只见得差别.
        
        说明最后的结论是他们2个相等.
        表示我经过网络得到的q值,和公式*算出来的结果是一样的.
        
        
        
     q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)  公式*
        
        还是跟q 学习一样的证明:
        假设只有3个state a,b,c 开始在b ,奖励是c
        那么b,right 的奖励是1.
        因为开始参数是0,所以结果是0,所以q_target(b,left)=1
        下面假设有5步 a,b,c,d,e
        开始在c,
        那么一样的.奖励从e传到d传到c.就得到了整体的q表.
        
        只不过,这里面不再存储q表了,而是存储生成q表的神经网络参数,就等价于存q表了.
        
        
        
注意点:q_next使用s_来计算. q_eval使用s来计算.
        这点对照q学习的公式 q_target = r + self.gamma * self.q_table.loc[s_, :].max() 
        来理解.这个公式的右边,需要的是将来步奏的q值.
        所以q_next使用的是s_来生成类似q学习里面的self.q_table.loc[s_, :].max()  这个数值.
        
        
        对比q学习里面公式:
        q_predict = self.q_table.loc[s, a]
        
        我们就知道了q_eval应该用当前步奏的state来传入也就是利用s来计算.
        
        整体loss就是那个bellman方程.当前q期望=当前奖励+未来q值.(只不过这个奖励值表示的是当前state,当前aciton
        做完之后会得到的奖励,奖励的定义也完全不影响算法.只要奖励定义的合理符合实际逻辑即可.).
        
        
        
        
        
        对于dqn里面创新点的理解:
        1.是q_next, q_eval 的不同步更新.q_next要每300步才从q_eval网络更新过来参数.
        因为从*式知道本质是reward.不是后面的q表.所以q表没必要每次都进行精细计算.
        所以300步更新一次就够了.误差不会变大,会节省接近1/2的算力.加速百分之百.
        
        
        
        
        
        
        '''











        """
        For example in this batch I have 2 samples and 3 actions:
        
        第一个sample 里面 action 1取值是1, action2 取值2, action3 取值3
        第2个sample 里面 action 1取值是4, action2 取值5, action3 取值6
        
        
        q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        q_target = q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        Then change q_target with the real q_target value w.r.t the q_eval's action.
        就是利用上面关键方程进行运算得到q_target
        For example in:
            sample 0, I took action 0, and the max q_target value is -1;
            sample 1, I took action 2, and the max q_target value is -2:
        q_target =
        [[-1, 2, 3],
         [4, 5, -2]]

        So the (q_target - q_eval) becomes:
        [[(-1)-(1), 0, 0],
         [0, 0, (-2)-(6)]]

        We then backpropagate this error w.r.t the corresponding action to network,
        leave other action as error=0 cause we didn't choose it.
        """

        # train eval network


        # 逆向bp算法即可.求loss
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})













        #用于画画用.也不是核心
        self.cost_his.append(self.cost)






#下面是epsilon 改进的组建增加策略,不是算法核心,用不用都行.用了能加速收敛.
        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()



