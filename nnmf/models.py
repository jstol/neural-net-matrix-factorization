#!/usr/bin/env python2.7
from __future__ import absolute_import, print_function
"""Defines NNMF models."""
# Third party modules
import tensorflow as tf
# Local modules
from .utils import KL, build_mlp, get_kl_weight

class _NNMFBase(object):
    def __init__(self, num_users, num_items, D=10, Dprime=60, hidden_units_per_layer=50,
                 latent_normal_init_params={'mean': 0.0, 'stddev': 0.1}, model_filename='model/nnmf.ckpt'):
        self.num_users = num_users
        self.num_items = num_items
        self.D = D
        self.Dprime = Dprime
        self.hidden_units_per_layer = hidden_units_per_layer
        self.latent_normal_init_params = latent_normal_init_params
        self.model_filename = model_filename

        # Internal counter to keep track of current iteration
        self._iters = 0

        # Input
        self.user_index = tf.placeholder(tf.int32, [None])
        self.item_index = tf.placeholder(tf.int32, [None])
        self.r_target = tf.placeholder(tf.float32, [None])

        # Call methods to initialize variables and operations (to be implemented by children)
        self._init_vars()
        self._init_ops()

        # RMSE
        self.rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(self.r, self.r_target))))

    def _init_vars(self):
        raise NotImplementedError

    def _init_ops(self):
        raise NotImplementedError

    def init_sess(self, sess):
        self.sess = sess
        init = tf.initialize_all_variables()
        self.sess.run(init)

    def _train_iteration(self, data, additional_feed=None):
        user_ids = data['user_id']
        item_ids = data['item_id']
        ratings = data['rating']

        feed_dict = {self.user_index: user_ids, self.item_index: item_ids, self.r_target: ratings}

        if additional_feed:
            feed_dict.update(additional_feed)

        for step in self.optimize_steps:
            self.sess.run(step, feed_dict=feed_dict)

        self._iters += 1

    def train_iteration(self, data):
        self._train_iteration(data)

    def eval_loss(self, data):
        raise NotImplementedError

    def eval_rmse(self, data):
        user_ids = data['user_id']
        item_ids = data['item_id']
        ratings = data['rating']

        feed_dict = {self.user_index: user_ids, self.item_index: item_ids, self.r_target: ratings}
        return self.sess.run(self.rmse, feed_dict=feed_dict)

    def predict(self, user_id, item_id):
        rating = self.sess.run(self.r, feed_dict={self.user_index: [user_id], self.item_index: [item_id]})
        return rating[0]

class NNMF(_NNMFBase):
    def __init__(self, *args, **kwargs):
        if 'lam' in kwargs:
            self.lam = float(kwargs['lam'])
            del kwargs['lam']
        else:
            self.lam = 0.01

        super(NNMF, self).__init__(*args, **kwargs)

    def _init_vars(self):
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

        # MLP ("f")
        f_input_layer = tf.concat(concat_dim=1, values=[self.U_lu, self.V_lu, tf.mul(self.Uprime_lu, self.Vprime_lu)])

        _r, self.mlp_weights = build_mlp(f_input_layer, hidden_units_per_layer=self.hidden_units_per_layer)
        self.r = tf.squeeze(_r, squeeze_dims=[1])

    def _init_ops(self):
        # Loss
        reconstruction_loss = tf.reduce_sum(tf.square(tf.sub(self.r_target, self.r)), reduction_indices=[0])
        reg = tf.add_n([tf.reduce_sum(tf.square(self.Uprime), reduction_indices=[0,1]),
                        tf.reduce_sum(tf.square(self.U), reduction_indices=[0,1]),
                        tf.reduce_sum(tf.square(self.V), reduction_indices=[0,1]),
                        tf.reduce_sum(tf.square(self.Vprime), reduction_indices=[0,1])])
        self.loss = reconstruction_loss + (self.lam*reg)

        # Optimizer
        self.optimizer = tf.train.AdamOptimizer()
        # Optimize the MLP weights
        f_train_step = self.optimizer.minimize(self.loss, var_list=self.mlp_weights.values())
        # Then optimize the latents
        latent_train_step = self.optimizer.minimize(self.loss, var_list=[self.U, self.Uprime, self.V, self.Vprime])

        self.optimize_steps = [f_train_step, latent_train_step]

    def eval_loss(self, data):
        user_ids = data['user_id']
        item_ids = data['item_id']
        ratings = data['rating']

        feed_dict = {self.user_index: user_ids, self.item_index: item_ids, self.r_target: ratings}
        return self.sess.run(self.loss, feed_dict=feed_dict)

class SVINNMF(_NNMFBase):
    num_latent_samples = 1
    num_data_samples = 3

    def __init__(self, *args, **kwargs):
        if 'r_var' in kwargs:
            self.r_var = float(kwargs['r_var'])
            del kwargs['r_sigma']
        else:
            self.r_var = 1.0

        if 'U_prior_var' in kwargs:
            self.U_prior_var = float(kwargs['U_prior_var'])
            del kwargs['U_prior_var']
        else:
            self.U_prior_var = 5.0

        if 'Uprime_prior_var' in kwargs:
            self.Uprime_prior_var = float(kwargs['Uprime_prior_var'])
            del kwargs['Uprime_prior_var']
        else:
            self.Uprime_prior_var = 5.0

        if 'V_prior_var' in kwargs:
            self.V_prior_var = float(kwargs['V_prior_var'])
            del kwargs['V_prior_var']
        else:
            self.V_prior_var = 5.0

        if 'Vprime_prior_var' in kwargs:
            self.Vprime_prior_var = float(kwargs['Vprime_prior_var'])
            del kwargs['Vprime_prior_var']
        else:
            self.Vprime_prior_var = 5.0

        if 'kl_full_iter' in kwargs:
            self.kl_full_iter = int(kwargs['kl_full_iter'])
            del kwargs['kl_full_iter']
        else:
            self.kl_full_iter = 1000 # something like: max_iters/3

        if 'anneal_kl' in kwargs:
            self.anneal_kl = bool(kwargs['anneal_kl'])
        else:
            self.anneal_kl = True

        super(SVINNMF, self).__init__(*args, **kwargs)

    def _init_vars(self):
        # Latents
        self.U_mu = tf.Variable(tf.truncated_normal(
            [self.num_users, self.D], **self.latent_normal_init_params))
        self.U_log_var = tf.Variable(tf.random_uniform(
            [self.num_users, self.D], minval=0.0, maxval=0.5))

        self.Uprime_mu = tf.Variable(tf.truncated_normal(
            [self.num_users, self.Dprime], **self.latent_normal_init_params))
        self.Uprime_log_var = tf.Variable(tf.random_uniform(
            [self.num_users, self.Dprime], minval=0.0, maxval=0.5))

        self.V_mu = tf.Variable(tf.truncated_normal(
            [self.num_items, self.D], **self.latent_normal_init_params))
        self.V_log_var = tf.Variable(tf.random_uniform(
            [self.num_items, self.D], minval=0.0, maxval=0.5))

        self.Vprime_mu = tf.Variable(tf.truncated_normal(
            [self.num_items, self.Dprime], **self.latent_normal_init_params))
        self.Vprime_log_var = tf.Variable(tf.random_uniform(
            [self.num_items, self.Dprime], minval=0.0, maxval=0.5))

        # Lookups
        U_mu_lu = tf.nn.embedding_lookup(self.U_mu, self.user_index)
        U_log_var_lu = tf.nn.embedding_lookup(self.U_log_var, self.user_index)

        Uprime_mu_lu = tf.nn.embedding_lookup(self.Uprime_mu, self.user_index)
        Uprime_log_var_lu = tf.nn.embedding_lookup(self.Uprime_log_var, self.user_index)

        V_mu_lu = tf.nn.embedding_lookup(self.V_mu, self.item_index)
        V_log_var_lu = tf.nn.embedding_lookup(self.V_log_var, self.item_index)

        Vprime_mu_lu = tf.nn.embedding_lookup(self.Vprime_mu, self.item_index)
        Vprime_log_var_lu = tf.nn.embedding_lookup(self.Vprime_log_var, self.item_index)

        # Posterior (q) - note this handles reparameterization for us
        q_U = tf.contrib.distributions.MultivariateNormalDiag(mu=U_mu_lu,
            diag_stdev=tf.sqrt(tf.exp(U_log_var_lu)))
        q_Uprime = tf.contrib.distributions.MultivariateNormalDiag(mu=Uprime_mu_lu,
            diag_stdev=tf.sqrt(tf.exp(Uprime_log_var_lu)))
        q_V = tf.contrib.distributions.MultivariateNormalDiag(mu=V_mu_lu,
            diag_stdev=tf.sqrt(tf.exp(V_log_var_lu)))
        q_Vprime = tf.contrib.distributions.MultivariateNormalDiag(mu=Vprime_mu_lu,
            diag_stdev=tf.sqrt(tf.exp(Vprime_log_var_lu)))

        # Sample
        self.U = q_U.sample()
        self.Uprime = q_Uprime.sample()
        self.V = q_V.sample()
        self.Vprime = q_Vprime.sample()

        # MLP ("f")
        f_input_layer = tf.concat(concat_dim=1, values=[self.U, self.V, tf.mul(self.Uprime, self.Vprime)])

        self.r_mu, self.mlp_weights = build_mlp(f_input_layer, hidden_units_per_layer=self.hidden_units_per_layer)
        self.r = tf.squeeze(self.r_mu, squeeze_dims=[1])

        # For KL annealing
        self.kl_weight = tf.placeholder(tf.float32) if self.anneal_kl else tf.constant(1.0, dtype=tf.float32)

    def _init_ops(self):
        KL_U = KL(self.U_mu, self.U_log_var, prior_var=self.U_prior_var)
        KL_Uprime = KL(self.Uprime_mu, self.Uprime_log_var, prior_var=self.Uprime_prior_var)
        KL_V = KL(self.V_mu, self.V_log_var, prior_var=self.V_prior_var)
        KL_Vprime = KL(self.Vprime_mu, self.Vprime_log_var, prior_var=self.Vprime_prior_var)
        KL_all = KL_U + KL_Uprime + KL_V + KL_Vprime

        # TODO weighting of gradient, handle multiple samples
        log_prob = -(1/(2.0*self.r_var))*tf.reduce_sum(tf.square(tf.sub(self.r_target, self.r)), reduction_indices=[0])
        elbo = log_prob-(self.kl_weight*KL_all)
        self.loss = -elbo

        self.optimizer = tf.train.AdamOptimizer()
        self.optimize_steps = [self.optimizer.minimize(self.loss)]

    def train_iteration(self, data):
        additional_feed = {self.kl_weight: get_kl_weight(self._iters, on_iter=self.kl_full_iter)} if self.anneal_kl \
            else {}
        super(SVINNMF, self)._train_iteration(data, additional_feed=additional_feed)

    def eval_loss(self, data):
        user_ids = data['user_id']
        item_ids = data['item_id']
        ratings = data['rating']

        feed_dict = {self.user_index: user_ids, self.item_index: item_ids, self.r_target: ratings,
                     self.kl_weight: get_kl_weight(self._iters, on_iter=self.kl_full_iter)}
        return self.sess.run(self.loss, feed_dict=feed_dict)
