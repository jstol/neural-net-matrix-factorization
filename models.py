#!/usr/bin/env python2.7
from __future__ import absolute_import, print_function
"""NNMF model."""
# Standard modules
from math import sqrt
# Third party modules
import tensorflow as tf

class _NNMFBase(object):
    def __init__(self, num_users, num_items, lam=0.01, learning_rate=0.001, D=10, Dprime=60, hidden_units_per_layer=50,
                 latent_normal_init_params={'mean': 0.0, 'stddev': 0.1}):
        self.num_users = num_users
        self.num_items = num_items
        self.lam = lam
        self.learning_rate = learning_rate
        self.D = D
        self.Dprime = Dprime
        self.hidden_units_per_layer = hidden_units_per_layer
        self.latent_normal_init_params = latent_normal_init_params

        self._init_vars()
        self._init_ops()

    def init_sess(self, sess):
        self.sess = sess
        init = tf.initialize_all_variables()
        self.sess.run(init)

    def _weight_init_range(self, n_in, n_out):
        range = 4.0 * sqrt(6.0) / sqrt(n_in + n_out)
        return {
            'minval': -range,
            'maxval': range,
        }

    def _build_mlp(self, f_input_layer):
        # TODO make number of hidden layers a parameter
        num_f_inputs = f_input_layer.get_shape().as_list()[1]

        # MLP weights picked uniformly from +/- 4*sqrt(6)/sqrt(n_in + n_out)
        mlp_weights = {
            'h1': tf.Variable(tf.random_uniform([num_f_inputs, self.hidden_units_per_layer],
                **self._weight_init_range(num_f_inputs, self.hidden_units_per_layer))),
            'h2': tf.Variable(tf.random_uniform([self.hidden_units_per_layer, self.hidden_units_per_layer],
                **self._weight_init_range(self.hidden_units_per_layer, self.hidden_units_per_layer))),
            'h3': tf.Variable(tf.random_uniform([self.hidden_units_per_layer, self.hidden_units_per_layer],
                **self._weight_init_range(self.hidden_units_per_layer, self.hidden_units_per_layer))),
            'out': tf.Variable(tf.random_uniform([self.hidden_units_per_layer, 1],
                **self._weight_init_range(self.hidden_units_per_layer, 1))),
        }
        # MLP layers
        mlp_layer_1 = tf.nn.sigmoid(tf.matmul(f_input_layer, mlp_weights['h1']))
        mlp_layer_2 = tf.nn.sigmoid(tf.matmul(mlp_layer_1, mlp_weights['h2']))
        mlp_layer_3 = tf.nn.sigmoid(tf.matmul(mlp_layer_2, mlp_weights['h3']))

        return tf.squeeze(tf.matmul(mlp_layer_3, mlp_weights['out']), squeeze_dims=[1]), mlp_weights

    def train_iteration(self, data):
        user_ids = data['user_id']
        item_ids = data['item_id']
        ratings = data['rating']

        feed_dict = {self.user_index: user_ids, self.item_index: item_ids, self.r_target: ratings}

        # Optimize weights
        self.sess.run(self.f_train_step, feed_dict=feed_dict)
        # Optimize latents
        self.sess.run(self.latent_train_step, feed_dict=feed_dict)

    def eval_rmse(self, data):
        user_ids = data['user_id']
        item_ids = data['item_id']
        ratings = data['rating']

        feed_dict = {self.user_index: user_ids, self.item_index: item_ids, self.r_target: ratings}
        rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(self.r, self.r_target))))
        return self.sess.run(rmse, feed_dict=feed_dict)

    def predict(self, user_id, item_id):
        rating = self.sess.run(self.r, feed_dict={self.user_index: [user_id], self.item_index: [item_id]})
        return rating[0]

class NNMF(_NNMFBase):
    model_filename = 'model/nnmf.ckpt'

    def _init_vars(self):
        # Input
        self.user_index = tf.placeholder(tf.int32, [None])
        self.item_index = tf.placeholder(tf.int32, [None])
        self.r_target = tf.placeholder(tf.float32, [None])

        # Latents
        self.U = tf.Variable(tf.truncated_normal([self.num_users, self.D], **self.latent_normal_init_params))
        self.Uprime = tf.Variable(tf.truncated_normal([self.num_users, self.Dprime], **self.latent_normal_init_params))
        self.V = tf.Variable(tf.truncated_normal([self.num_items, self.D], **self.latent_normal_init_params))
        self.Vprime = tf.Variable(tf.truncated_normal([self.num_items, self.Dprime], **self.latent_normal_init_params))

        # Lookups
        self.U_lu = tf.nn.embedding_lookup(self.U, self.user_index)
        self.Uprime_lu = tf.nn.embedding_lookup(self.Uprime, self.user_index)
        self.V_lu = tf.nn.embedding_lookup(self.V, self.item_index)
        self.Vprime_lu = tf.nn.embedding_lookup(self.Vprime, self.item_index)

        # MLP ("f") - TODO make this nicer, if possible (loop?)
        self.f_input_layer = tf.concat(concat_dim=1,
                                       values=[self.U_lu, self.V_lu, tf.mul(self.Uprime_lu, self.Vprime_lu)])

        self.r, self.mlp_weights = self._build_mlp(self.f_input_layer)

    def _init_ops(self):
        # Loss
        reconstruction_loss = tf.reduce_sum(tf.square(tf.sub(self.r_target, self.r)))
        reg = tf.add_n([tf.nn.l2_loss(self.Uprime), tf.nn.l2_loss(self.U),
                        tf.nn.l2_loss(self.V), tf.nn.l2_loss(self.Vprime)])
        self.loss = tf.add(reconstruction_loss, tf.scalar_mul(self.lam, reg))

        # Optimizer
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, name='opt')
        self.f_train_step = self.optimizer.minimize(self.loss, var_list=self.mlp_weights.values())
        self.latent_train_step = self.optimizer.minimize(self.loss, var_list=[self.U, self.Uprime, self.V, self.Vprime])
