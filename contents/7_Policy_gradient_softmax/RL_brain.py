"""
This part of code is the reinforcement learning brain, which is a brain of the agent.
All decisions are made in here.

Policy Gradient, Reinforcement Learning.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""

import numpy as np
import tensorflow as tf

# reproducible
np.random.seed(1)
tf.set_random_seed(1)

'''
这种算法是每一次学习action的选择概率.用神经网络来输出action对应的概率值.
所以:
dqn:用于state无限多,action少的情况
pg:用于state少,action无限多的情况.
如果state,action都有限,那么问题比较简单,2个方法都试用.
'''
class PolicyGradient:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.95,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        self._build_net()

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # http://0.0.0.0:6006/
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        with tf.name_scope('inputs'):
            self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name="observations")
            self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num")
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")

        #     通过衰减的bellman公式,就能把这个tf_vt 也就是当前步奏的奖励传递给,action表中的每一个历史过程
        # 从而渐进性逆向传导.
        # fc1
        layer = tf.layers.dense(
            inputs=self.tf_obs,
            units=10,
            activation=tf.nn.tanh,  # tanh activation
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc1'
        )
        # fc2
        all_act = tf.layers.dense(
            inputs=layer,
            units=self.n_actions,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc2'
        )
        # all_act 表示输出所有动作得分预测 的这个向量值.
        self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')  # use softmax to convert to probability

        with tf.name_scope('loss'):
            # to maximize total reward (log_p * R) is to minimize -(log_p * R), and the tf only have minimize(loss)


            #   self.tf_acts   是 真实的奖励结果.
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.tf_acts)   # this is negative log of chosen action
            # 理解上面这一行是关键:   神经网络输出的是预测出来的分布较all_act,根据预测出来的value实际上走的acts是
            # self.tf_acts. 他们的 差别用 croos_entropy来衡量. 那么这个差别是好还是坏呢, 我们乘以最后的reward.
            #也就是self.tf_vt.   他代表每一步走的好坏.平均下来后,让loss最小.就表示
            # 如果奖励是负的,那么loss极小化就等价于交叉熵也就是让分布变得不像.反之也显然.




            # or in this way:   下面一行就是使用交叉熵公式 :交叉熵= -log预测的  *真实的. 计算两个分布的差别时候肯定用这个,然后极小化这个东西就是loss函数了.
            #证明交叉熵公式就是极大似然法就行了. 这点可以参考逻辑回归里面的loss函数,他们很像.
            # kl散度就是相对熵
            # https://www.zhihu.com/question/65288314 交叉熵的计算.
            # neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)
            loss = tf.reduce_mean(neg_log_prob * self.tf_vt)  # reward guided loss

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def choose_action(self, observation):#根据自己网络生成的概率进行choice动作.进行游戏模拟.
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation[np.newaxis, :]})
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)     # 观测,  动作,  奖励  存起来.
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):
        # discount and normalize episode reward          #先把真实的奖励进行衰减分配.这样就得到了真实的value表.
        discounted_ep_rs_norm = self._discount_and_norm_rewards()
#  discounted_ep_rs_norm 这个就是从现实模型反馈回来的价值表.做逆向bp算法即可.
        # train on episode
        self.sess.run(self.train_op, feed_dict={
             self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
             self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
             self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
        })

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []    # empty episode data
        return discounted_ep_rs_norm

    def _discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):         #大概是这个衰减概念.   #  ep_rs:  a,b,c,d
                                                                #  discounted_ep_rs:  a+b*0.2+c*0.1+d*0.01,b+d*0.2+c*0.1,c+d*0.22,d
                                #就是  根据奖励公式计算的.


            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs



