#!/usr/bin/env python2.7
from __future__ import absolute_import, print_function
"""Runs NNMF model."""
# Standard modules
from math import sqrt
# Third party modules
import tensorflow as tf
import tensorflow.contrib.bayesflow as bf

class _NNMFBase(object):
    def __init__(self, num_users, num_items, lam=0.01, learning_rate=0.001, D=10, Dprime=60, hidden_units_per_layer=50,
                 latent_normal_init_params={'mean': 0.0, 'stddev': 0.1}, model_filename='model/nnmf.ckpt'):
        self.num_users = num_users
        self.num_items = num_items
        self.lam = lam
        self.learning_rate = learning_rate
        self.D = D
        self.Dprime = Dprime
        self.hidden_units_per_layer = hidden_units_per_layer
        self.latent_normal_init_params = latent_normal_init_params
        self.model_filename = model_filename

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
        self.mlp_weights = {
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
        self.mlp_layer_1 = tf.nn.sigmoid(tf.matmul(f_input_layer, self.mlp_weights['h1']))
        self.mlp_layer_2 = tf.nn.sigmoid(tf.matmul(self.mlp_layer_1, self.mlp_weights['h2']))
        self.mlp_layer_3 = tf.nn.sigmoid(tf.matmul(self.mlp_layer_2, self.mlp_weights['h3']))
        self.out = tf.matmul(self.mlp_layer_3, self.mlp_weights['out'])

        return self.out, self.mlp_weights

    def train_iteration(self, data):
        user_ids = data['user_id']
        item_ids = data['item_id']
        ratings = data['rating']

        feed_dict = {self.user_index: user_ids, self.item_index: item_ids, self.r_target: ratings}

        for step in self.optimize_steps:
            self.sess.run(step, feed_dict=feed_dict)

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

        _r, self.mlp_weights = self._build_mlp(self.f_input_layer)
        self.r = tf.squeeze(_r, squeeze_dims=[1])

    def _init_ops(self):
        # Loss
        reconstruction_loss = tf.reduce_sum(tf.square(tf.sub(self.r_target, self.r)))
        reg = tf.add_n([tf.nn.l2_loss(self.Uprime), tf.nn.l2_loss(self.U),
                        tf.nn.l2_loss(self.V), tf.nn.l2_loss(self.Vprime)])
        self.loss = tf.add(reconstruction_loss, tf.scalar_mul(self.lam, reg))

        # Optimizer
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
        # Optimize the MLP weights
        f_train_step = self.optimizer.minimize(self.loss, var_list=self.mlp_weights.values())
        # Then optimize the latents
        latent_train_step = self.optimizer.minimize(self.loss, var_list=[self.U, self.Uprime, self.V, self.Vprime])
        self.optimize_steps = [f_train_step, latent_train_step]

class SVINNMF(_NNMFBase):
    num_latent_samples = 1
    num_data_samples = 3

    def __init__(self, *args, **kwargs):
        if 'r_sigma' in kwargs:
            self.r_sigma = kwargs['r_sigma']
            del kwargs['r_sigma']
        else:
            self.r_sigma = 0.75

        super(SVINNMF, self).__init__(*args, **kwargs)

    def _init_vars(self):
        # Input
        self.user_index = tf.placeholder(tf.int32, [None])
        self.item_index = tf.placeholder(tf.int32, [None])
        self.r_target = tf.placeholder(tf.float32, [None])

        # Latents
        self.U_mu = tf.Variable(tf.truncated_normal(
            [self.num_users, self.D], **self.latent_normal_init_params))
        self.U_sigma = tf.Variable(tf.random_uniform(
            [self.num_users, self.D], minval=0.0, maxval=1.0))

        self.Uprime_mu = tf.Variable(tf.truncated_normal(
            [self.num_users, self.Dprime], **self.latent_normal_init_params))
        self.Uprime_sigma = tf.Variable(tf.random_uniform(
            [self.num_users, self.Dprime], minval=0.0, maxval=1.0))

        self.V_mu = tf.Variable(tf.truncated_normal(
            [self.num_items, self.D], **self.latent_normal_init_params))
        self.V_sigma = tf.Variable(tf.random_uniform(
            [self.num_items, self.D], minval=0.0, maxval=1.0))

        self.Vprime_mu = tf.Variable(tf.truncated_normal(
            [self.num_items, self.Dprime], **self.latent_normal_init_params))
        self.Vprime_sigma = tf.Variable(tf.random_uniform(
            [self.num_items, self.Dprime], minval=0.0, maxval=1.0))

        # Lookups
        self.U_mu_lu = tf.nn.embedding_lookup(self.U_mu, self.user_index)
        self.U_sigma_lu = tf.nn.embedding_lookup(self.U_sigma, self.user_index)

        self.Uprime_mu_lu = tf.nn.embedding_lookup(self.Uprime_mu, self.user_index)
        self.Uprime_sigma_lu = tf.nn.embedding_lookup(self.Uprime_sigma, self.user_index)

        self.V_mu_lu = tf.nn.embedding_lookup(self.V_mu, self.item_index)
        self.V_sigma_lu = tf.nn.embedding_lookup(self.V_sigma, self.item_index)

        self.Vprime_mu_lu = tf.nn.embedding_lookup(self.Vprime_mu, self.item_index)
        self.Vprime_sigma_lu = tf.nn.embedding_lookup(self.Vprime_sigma, self.item_index)

        # priors
        self.p_V = self.p_U = tf.contrib.distributions.MultivariateNormalDiag(mu=tf.zeros(shape=[self.D]),
                                                              diag_stdev=tf.ones([self.D]))

        self.p_Vprime = self.p_Uprime = tf.contrib.distributions.MultivariateNormalDiag(mu=tf.zeros(shape=[self.Dprime]),
                                                                        diag_stdev=tf.ones([self.Dprime]))

        # Posterior (q)
        self.q_U = tf.contrib.distributions.MultivariateNormalDiag(mu=self.U_mu_lu, diag_stdev=self.U_sigma_lu)
        self.q_Uprime = tf.contrib.distributions.MultivariateNormalDiag(mu=self.Uprime_mu_lu, diag_stdev=self.Uprime_sigma_lu)
        self.q_V = tf.contrib.distributions.MultivariateNormalDiag(mu=self.V_mu_lu, diag_stdev=self.V_sigma_lu)
        self.q_Vprime = tf.contrib.distributions.MultivariateNormalDiag(mu=self.Vprime_mu_lu, diag_stdev=self.Vprime_sigma_lu)

        # Sample
        # TODO not just mean and more than one sample
        self.U = self.q_U.mean()
        self.Uprime = self.q_Uprime.mean()
        self.V = self.q_V.mean()
        self.Vprime = self.q_Vprime.mean()

        # MLP ("f")
        self.f_input_layer = tf.concat(concat_dim=1,
                                       values=[self.U, self.V, tf.mul(self.Uprime, self.Vprime)])

        self.r_mu, self.mlp_weights = self._build_mlp(self.f_input_layer)

        # TODO learn sigma and take it into account in loss
        self.p_r_given_Z = tf.contrib.distributions.Normal(mu=self.r_mu, sigma=self.r_sigma*tf.ones_like(self.r_mu))

    def _init_ops(self):
        # TODO do i have to tile p_U?
        # self.KL_U = tf.contrib.distributions.kl(self.q_U, self.p_U)
        # self.KL_Uprime = tf.contrib.distributions.kl(self.q_Uprime, self.p_Uprime)
        # self.KL_V = tf.contrib.distributions.kl(self.q_V, self.p_V)
        # self.KL_Vprime = tf.contrib.distributions.kl(self.q_Vprime, self.p_Vprime)
        # self.KL_all = tf.reduce_sum(self.KL_U + self.KL_Uprime + self.KL_V + self.KL_Vprime, reduction_indices=[0])

        # TODO weighting of gradient
        # TODO handle multiple samples
        # self.r_target_stacked = tf.squeeze(tf.reshape(tf.tile(tf.expand_dims(self.r_target, 1), [1, self.num_latent_samples]),
        #                               [self.num_data_samples * self.num_latent_samples, 1]), squeeze_dims=[1])

        self.log_prob = self.p_r_given_Z.log_pdf(tf.transpose([self.r_target]))
        self.loss = -tf.reduce_sum(self.log_prob)

        # self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
        self.optimizer = tf.train.AdamOptimizer()
        self.optimize_steps = [self.optimizer.minimize(self.loss)]

    def eval_rmse(self, data):
        user_ids = data['user_id']
        item_ids = data['item_id']
        ratings = data['rating']

        feed_dict = {self.user_index: user_ids, self.item_index: item_ids, self.r_target: ratings}

        return self.sess.run((self.r_mu, -self.log_prob, self.loss), feed_dict=feed_dict)

    def predict(self, user_id, item_id):
        rating = self.sess.run(self.r_mu, feed_dict={self.user_index: [user_id], self.item_index: [item_id]})
        return rating