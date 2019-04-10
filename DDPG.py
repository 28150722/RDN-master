"""
Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.
DDPG is Actor Critic based algorithm.
Pendulum example.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
tensorflow 1.0
gym 0.8.0
"""
import tensorflow as tf
FLAGS = tf.app.flags.FLAGS
import tensorflow as tf


import data_provider
import numpy as np
import rdn_model
import pdb
import os.path
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

num_patches = 68
# num_patches = 19
k_nearest = 4
np.random.seed(1)
tf.set_random_seed(1)


def tf_normalized_rmse(pred, gt_truth):
    norm = tf.sqrt(tf.reduce_sum(((gt_truth[:, 36, :] - gt_truth[:, 45, :]) ** 2), 1))  # out-ocular distance

    return tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(pred - gt_truth), 2)), 1) / (norm * 68)


###############################  Actor  ####################################


class Actor(object):
    def __init__(self, sess, shape_space, k, learning_rate, replacement):
        self.sess = sess
        self.image = data_provider.distort_color(image_holder)
        self.shape_space = shape_space
        self.k = k
        # self.a_dim = action_dim
        # self.action_bound = action_bound
        self.lr = learning_rate
        self.replacement = replacement
        self.t_replace_counter = 0
        self.k_nn = K_nn(shape_space, self.k)
        self.find_k_nn = self.k_nn.find_k_nn(S)
        self.find_k_nn_target = self.k_nn.find_k_nn(S_)

        with tf.variable_scope('Actor'):
            # input s, output a
            self.a, self.dxs = self._build_net(S, scope='eval_net', trainable=True)

            # input s_, output a, get a_ for critic
            # self.a_, self.dxs_ = self._build_net(S_, scope='target_net', trainable=False)

        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval_net')
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target_net')

        if self.replacement['name'] == 'hard':
            self.t_replace_counter = 0
            self.hard_replace = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]
        else:
            self.soft_replace = [tf.assign(t, (1 - self.replacement['tau']) * t + self.replacement['tau'] * e)
                                 for t, e in zip(self.t_params, self.e_params)]

        self.loss_supervised = tf.reduce_mean(tf_normalized_rmse(self.a + S, shape_gt))
        opt = tf.train.AdamOptimizer(self.lr)
        gvs = opt.compute_gradients(self.loss_supervised)
        capped_gvs = []
        for grad, var in gvs:
            if grad != None and (var in self.e_params):
                capped_gvs.append((tf.clip_by_value(grad, -0.1, 0.1), var))
        self.train_op_supervised = opt.apply_gradients(capped_gvs)
        self.grads_supervised = tf.gradients(self.loss_supervised, self.a)

    def _build_net(self, s, scope, trainable):
        with tf.variable_scope(scope):
            # init_w = tf.random_normal_initializer(0., 0.3)
            # init_b = tf.constant_initializer(0.1)
            # net = tf.layers.dense(s, 30, activation=tf.nn.relu,
            #                       kernel_initializer=init_w, bias_initializer=init_b, name='l1',
            #                       trainable=trainable)
            # with tf.variable_scope('a'):
            #     actions = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, kernel_initializer=init_w,
            #                               bias_initializer=init_b, name='a', trainable=trainable)
            #     scaled_a = tf.multiply(actions, self.action_bound, name='scaled_a')  # Scale output to -action_bound to action_bound
            with tf.device('/gpu:0'):
                action, dxs, _ = rdn_model.model_actor(self.image, s, num_iterations=3, num_patches=num_patches,
                                                       patch_shape=(30, 30))
        return action, dxs

    def learn_supervise(self, s, gt, image):
        with tf.device('/gpu:0'):
            self.sess.run(self.train_op_supervised, feed_dict={S: s, shape_gt: gt, image_holder: image})

            return self.sess.run(self.grads_supervised,
                                 feed_dict={S: s,shape_gt: gt, image_holder: image}), self.sess.run(
                self.loss_supervised, feed_dict={S: s,shape_gt: gt, image_holder: image})

    # def learn(self, s, a_hat, image):  # batch update
    #     with tf.device('/gpu:0'):
    #         self.sess.run(self.train_op, feed_dict={S: s, q_a: a_hat, image_holder: image})
    #
    #         if self.replacement['name'] == 'soft':
    #             self.sess.run(self.soft_replace)
    #         else:
    #             if self.t_replace_counter % self.replacement['rep_iter_a'] == 0:
    #                 self.sess.run(self.hard_replace)
    #             self.t_replace_counter += 1

    def choose_action_hat(self, s, image):
        with tf.device('/gpu:0'):
            return self.sess.run(self.a, feed_dict={S: s,image_holder: image})

    def choose_dxs(self, s, image):
        with tf.device('/gpu:0'):
            return self.sess.run(self.dxs, feed_dict={S: s, image_holder: image})

    def choose_action(self, s, a, image):
        with tf.device('/gpu:0'):
            return self.sess.run(self.find_k_nn, feed_dict={S: s + a, image_holder: image})  # single action

    # def choose_action_target(self, s_, a_, image):
    #     with tf.device('/gpu:0'):
    #         # FUCK! BUT WHY???
    #         # s = s[np.newaxis, :]    # single state
    #         # k_nn_a = self.k_nn.find_k_nn(s_)
    #         # print('choose_action_target:', (s_).get_shape().as_list())
    #         return self.sess.run(self.find_k_nn_target, feed_dict={S_: s_ + a_, image_holder: image})  # single action

    # def add_grad_to_graph(self, a_grads):
    #
    #     # self.loss = q
    #     # self.grads_RL = tf.gradients(self.loss, self.a)
    #     # opt = tf.train.AdamOptimizer(self.lr)
    #     # gvs = opt.compute_gradients(self.loss)
    #     # capped_gvs = []
    #     # for grad, var in gvs:
    #     #     if grad != None and (var in self.e_params):
    #     #         capped_gvs.append((tf.clip_by_value(grad, -0.1, 0.1), var))
    #     # print('capped_gvs:', capped_gvs)
    #     # self.train_op = opt.apply_gradients(capped_gvs)
    #     with tf.device('/gpu:0'):
    #         with tf.variable_scope('policy_grads'):
    #             # ys = policy;
    #             # xs = policy's parameters;
    #             # self.a_grads = the gradients of the policy to get more Q
    #             # tf.gradients will calculate dys/dxs with a initial gradients for ys, so this is dq/da * da/dparams
    #             self.policy_grads = tf.gradients(ys=self.a, xs=self.e_params, grad_ys=a_grads)
    #
    #         with tf.variable_scope('A_train'):
    #             opt = tf.train.AdamOptimizer(self.lr)  # (- learning rate) for ascent policy
    #             self.train_op = opt.apply_gradients(zip(self.policy_grads, self.e_params))


###############################  Critic  ####################################

class Critic(object):
    def __init__(self, sess, learning_rate, gamma, replacement, k=k_nearest):
        self.sess = sess
        self.image = image_holder
        # self.s_dim = state_dim
        # self.a_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.replacement = replacement
        self.k = k

        with tf.variable_scope('Critic'):
            # Input (s, a), output q
            self.a = q_a
            self.a_ = q_a_
            self.q = self._build_net(S, self.a, 'eval_net', trainable=True)

            # Input (s_, a_), output q_ for q_target
            self.q_ = self._build_net(S_, self.a_, 'target_net',
                                      trainable=False)  # target_q is based on a_ from Actor's target_net
            # self.q_hat = self._build_net(S, a_hat, 'gradient_net', trainable=True)
            self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval_net')
            self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target_net')

        with tf.variable_scope('target_q'):
            self.target_q = R + self.gamma * self.q_

        with tf.variable_scope('TD_error'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.target_q, self.q))

        with tf.variable_scope('supervise'):
            self.targ = target
            self.loss_supervised = tf.reduce_mean(tf.squared_difference(self.q, self.targ))
            opt = tf.train.AdamOptimizer(self.lr)
            gvs = opt.compute_gradients(self.loss_supervised)
            # capped_gvs = []
            # for grad, var in gvs:
            #     if grad != None:
            #         capped_gvs.append((tf.clip_by_value(grad, -0.1, 0.1), var))
            # capped_gvs = [(tf.clip_by_value(grad, -1, 1), var) if grad != None, for grad, var in gvs]
            self.train_supervised_op = opt.apply_gradients(gvs)

        # with tf.variable_scope('C_train'):
        #     opt = tf.train.AdamOptimizer(self.lr)
        #     gvs = opt.compute_gradients(self.loss)
        #     capped_gvs = []
        #     for grad, var in gvs:
        #         if grad != None:
        #             capped_gvs.append((tf.clip_by_value(grad, -0.1, 0.1), var))
        #     # capped_gvs = [(tf.clip_by_value(grad, -1, 1), var) if grad != None, for grad, var in gvs]
        #     self.train_op = opt.apply_gradients(capped_gvs)

        with tf.variable_scope('a_grad'):
            self.a_grads = tf.gradients(self.q, self.a, (tf.sign(self.q - 0.12) + 1) / 2)[
                0]  # tensor of gradients of each sample (None, a_dim)
            # print self.a_grads

        if self.replacement['name'] == 'hard':
            self.t_replace_counter = 0
            self.hard_replacement = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]
        else:
            self.soft_replacement = [tf.assign(t, (1 - self.replacement['tau']) * t + self.replacement['tau'] * e)
                                     for t, e in zip(self.t_params, self.e_params)]

    def _build_net(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            # init_w = tf.random_normal_initializer(0., 0.1)
            # init_b = tf.constant_initializer(0.1)

            # with tf.variable_scope('l1'):
            #     n_l1 = 30
            #     w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], initializer=init_w, trainable=trainable)
            #     w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], initializer=init_w, trainable=trainable)
            #     b1 = tf.get_variable('b1', [1, n_l1], initializer=init_b, trainable=trainable)
            #     net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)

            # with tf.variable_scope('q'):
            #     q = tf.layers.dense(net, 1, kernel_initializer=init_w, bias_initializer=init_b, trainable=trainable)   # Q(s,a)
            print('Q value net: s, a shape:', s.get_shape().as_list(), a.get_shape().as_list())
            with tf.device('/gpu:0'):
                shape = s + a
                q = rdn_model.model_critic(self.image, shape, num_patches=num_patches)
                # if scope == 'eval_net':
                #     actor_grad = tf.gradients(actor.a, actor.e_params, grad_ys=a - actor.a)
                # if scope == 'target_net':
                #     actor_grad = tf.gradients(actor.a_, actor.t_params, grad_ys=a - actor.a)
                # init_w = tf.random_normal_initializer(0., 0.1)
                # for i, grad in enumerate(actor_grad):
                #     grad = tf.reshape(grad, [-1])
                #     w1 = tf.get_variable('w'+str(i), [grad.shape, n_l1], initializer=init_w, trainable=True)
        return q

    # def learn(self, s, a, r, s_, a_, image):
    #     with tf.device('/gpu:0'):
    #         # print('target q:', self.sess.run(self.target_q, feed_dict={S: s, q_a: a, R: r, S_: s_, q_a_: a_, image_holder:image}))
    #         # print('q:', self.sess.run(self.q, feed_dict={S: s, q_a: a, R: r, S_: s_, q_a_: a_, image_holder:image}))
    #         a_zeros = np.zeros(a.shape)
    #         self.sess.run(self.train_op,
    #                       feed_dict={S: s, q_a: a_zeros, R: r, S_: s + a, q_a_: a_zeros, image_holder: image})
    #         if self.replacement['name'] == 'soft':
    #             self.sess.run(self.soft_replacement)
    #         else:
    #             if self.t_replace_counter % self.replacement['rep_iter_c'] == 0:
    #                 self.sess.run(self.hard_replacement)
    #             self.t_replace_counter += 1
    #         return self.sess.run(self.loss, feed_dict={S: s, q_a: a, R: r, S_: s_, q_a_: a_, image_holder: image})

    def learn_supervised(self, s, a, image, gt):
        norm = np.sqrt(np.sum(((gt[:, 36, :] - gt[:, 45, :]) ** 2), 1))
        error = np.sum(np.sqrt(np.sum(np.square(s + a - gt), 2)), 1) / (norm * 68)
        # targ = error.reshape([-1, 68, 1])
        targ = np.expand_dims(error, -1)
        # targ = np.reshape(0.5 * (np.sign(error-0.1) + np.sign(error-0.05)), [-1,1])
        # print('error:', error)
        # print('targ:', targ)
        # print('q:', self.sess.run(self.q, feed_dict={S:s, q_a:a, image_holder:image}))
        loss = self.sess.run(self.loss_supervised, feed_dict={S: s, q_a: a, image_holder: image, target: targ})
        with tf.device('/gpu:0'):
            self.sess.run(self.train_supervised_op, feed_dict={S: s, q_a: a, image_holder: image, target: targ})
        return loss

    def q_value(self, s, a, image):
        with tf.device('/gpu:0'):
            # return (np.sum(self.sess.run(self.q, feed_dict={S:s, q_a:a, image_holder:image}), 1) / 68)
            return (self.sess.run(self.q
                                  , feed_dict={S: s, q_a: a, image_holder: image}))

    def a_grad(self, s, a, image):
        with tf.device('/gpu:0'):
            return self.sess.run(self.a_grads, feed_dict={S: s, q_a: a, image_holder: image})

    def get_a_grads(self, s, a, image):
        with tf.device('/gpu:0'):
            return self.sess.run(self.a_grads, feed_dict={S: s, q_a: a, image_holder: image})

    def choose_max(self, s, k_nn_a, image):
        with tf.device('/gpu:0'):
            q_list = []
            for i in range(self.k):
                # np.sum(np.sqrt(np.sum(np.square(pred - gt_truth), 2)), 1) / (norm * 68)
                q_list.append(np.sum(self.sess.run(self.q, feed_dict={S: np.zeros(s.shape), q_a: np.expand_dims(k_nn_a[i, :, :], 0),
                                                                      image_holder: image}), 1) / 68)
            # print('q_list:', q_list)
            max_index = q_list.index(min(q_list))
            # print('max_index:',max_index)
            s_chose = np.expand_dims(k_nn_a[max_index, :, :], 0)
            # flag = 0
            # while (rdn_model.normalized_rmse(s_chose, s) < 0.0005 or self.q_value(s, s_chose-s, image) > 1) and flag < k_nearest-1:
            #     q_list[max_index] = -1
            #     max_index = q_list.index(max(q_list))
            #     s_chose = np.expand_dims(k_nn_a[max_index, :, :], 0)
            #     flag += 1

            return s_chose - s

    # def choose_max_reward(self, s, k_nn_a, gt_shape):
    #     with tf.device('/gpu:0'):
    #         q_list = []
    #         for i in range(self.k):
    #             # np.sum(np.sqrt(np.sum(np.square(pred - gt_truth), 2)), 1) / (norm * 68)
    #             # q_list.append(np.sum(self.sess.run(self.q, feed_dict={S:s, q_a:np.expand_dims(k_nn_a[i,:,:],0), image_holder:image}),1) / 68)
    #             q_list.append(rdn_model.step(self.sess, np.expand_dims(k_nn_a[i, :, :], 0), s, gt_shape)[1])
    #         # pdb.set_trace()
    #         # print('q_list:' q_list)
    #         max_index = q_list.index(min(q_list))
    #         # print('max_index:',max_index)
    #         s_chose = np.expand_dims(k_nn_a[max_index, :, :], 0)
    #         # flag = 0
    #         # while (rdn_model.normalized_rmse(s_chose, s) < 0.0005 or self.q_value(s, s_chose-s, image) > 1) and flag < k_nearest-1:
    #         #     q_list[max_index] = -1
    #         #     max_index = q_list.index(max(q_list))
    #         #     s_chose = np.expand_dims(k_nn_a[max_index, :, :], 0)
    #         #     flag += 1
    #
    #         return s_chose - s


#####################  Memory  ####################

class Memory(object):
    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.memory_s = np.zeros((capacity, dims, 2))
        self.memory_a = np.zeros((capacity, dims, 2))
        self.memory_s_ = np.zeros((capacity, dims, 2))
        self.memory_r = np.zeros((capacity, 1))
        self.memory_idx = np.zeros((capacity, 1))
        self.memory_trans_func = range(capacity)
        self.memory_shape_gt = np.zeros((capacity, dims, 2))
        self.pointer = 0

    def store_transition(self, s, a, r, s_, idx, trans_func, shape_gt):
        # transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.capacity  # replace the old memory with new memory
        self.memory_s[index, :, :] = s
        # print('s:', s)
        self.memory_a[index, :, :] = a
        # print('s_:', s_)
        self.memory_s_[index, :, :] = s_
        # print('r:',r)
        self.memory_r[index, :] = r
        self.memory_idx[index] = idx
        self.memory_trans_func[index] = trans_func
        self.memory_shape_gt[index, :, :] = shape_gt
        self.pointer += 1

    def sample(self, n):
        assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
        indices = np.random.choice(self.capacity, size=n)
        trans_func_list = []
        for i in indices:
            trans_func_list.append(self.memory_trans_func[i])
        return self.memory_s[indices, :, :], self.memory_a[indices, :, :], self.memory_s_[indices, :, :], self.memory_r[
                                                                                                          indices, :], \
               self.memory_idx[indices, :], trans_func_list, self.memory_shape_gt[indices, :, :]

    def reset(self):
        capacity = self.capacity
        dims = num_patches
        self.memory_s = np.zeros((capacity, dims, 2))
        self.memory_a = np.zeros((capacity, dims, 2))
        self.memory_s_ = np.zeros((capacity, dims, 2))
        self.memory_r = np.zeros((capacity, 1))
        self.memory_idx = np.zeros((capacity, 1))
        self.memory_trans_func = range(capacity)
        self.pointer = 0


class K_nn(object):
    def __init__(self, shape_space, k):
        self.shape_space = shape_space
        self.k = k

    def normalized_rmse(self, shape_hat):
        error = []
        for i in range(self.shape_space.shape[0]):
            shape = tf.reshape(tf.convert_to_tensor(self.shape_space[i, :, :], dtype=tf.float32), [1, -1, 2])
            norm = tf.sqrt(tf.reduce_sum(((shape[:, 36, :] - shape[:, 45, :]) ** 2), 1))  # out-ocular distance
            error.append(tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(shape_hat - shape), 2)), 1) / (norm * 68))
        return tf.convert_to_tensor(error)

    def find_k_nn(self, shape_hat):
        error = tf.transpose(self.normalized_rmse(shape_hat), [1, 0])
        # print('error shape:', error.get_shape().as_list())
        # print('k:', self.k)
        min_index = tf.nn.top_k(-error, self.k).indices
        # print('min_index:', min_index)
        k_nn_shapes = tf.nn.embedding_lookup(self.shape_space, min_index)
        # print('k_nn_shapes :', k_nn_shapes.get_shape().as_list())
        return tf.squeeze(tf.cast(k_nn_shapes, tf.float32), [0])[:, :, :]


with tf.name_scope('S'):
    S = tf.placeholder(tf.float32, shape=[None, num_patches, 2], name='s')
with tf.name_scope('State_3D'):
    State_3D = tf.placeholder(tf.float32, shape=[None, 168], name='state')
with tf.name_scope('R'):
    R = tf.placeholder(tf.float32, [None, 1], name='r')
with tf.name_scope('S_'):
    S_ = tf.placeholder(tf.float32, shape=[None, num_patches, 2], name='s_')
with tf.name_scope('image'):
    image_holder = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='image')  # 386, 458   383, 453   272,261
with tf.name_scope('q_a'):
    q_a = tf.placeholder(tf.float32, shape=[None, num_patches, 2], name='q_a')
with tf.name_scope('q_a_'):
    q_a_ = tf.placeholder(tf.float32, shape=[None, num_patches, 2], name='q_a_')

with tf.name_scope('target'):
    target = tf.placeholder(tf.float32, shape=[None, 1], name='target')

with tf.name_scope('shape_gt'):
    shape_gt = tf.placeholder(tf.float32, shape=[None, num_patches, 2], name='shape_gt')