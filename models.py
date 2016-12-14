#!/usr/bin/env python2.7
from __future__ import absolute_import, print_function
"""Runs NNMF model."""
# Standard modules
import math
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

        # Internal counter to keep track of current iter
        self._iters = 0

        self._init_vars()
        self._init_ops()

    def init_sess(self, sess):
        self.sess = sess
        init = tf.initialize_all_variables()
        self.sess.run(init)

    def _weight_init_range(self, n_in, n_out):
        range = 4.0 * math.sqrt(6.0) / math.sqrt(n_in + n_out)
        return {
            'minval': -range,
            'maxval': range,
        }

    def _build_mlp(self, f_input_layer):
        # TODO look at tf.contrib.layers
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

    def train_iteration(self, data, additional_feed=None):
        user_ids = data['user_id']
        item_ids = data['item_id']
        ratings = data['rating']

        feed_dict = {self.user_index: user_ids, self.item_index: item_ids, self.r_target: ratings}

        if additional_feed:
            feed_dict.update(additional_feed)

        for step in self.optimize_steps:
            self.sess.run(step, feed_dict=feed_dict)

        self._iters += 1

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

        self.rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(self.r, self.r_target))))

    def _init_ops(self):
        # Loss
        reconstruction_loss = tf.reduce_sum(tf.square(tf.sub(self.r_target, self.r)), reduction_indices=[0])
        # reg = tf.add_n([tf.nn.l2_loss(self.Uprime), tf.nn.l2_loss(self.U),
        #                 tf.nn.l2_loss(self.V), tf.nn.l2_loss(self.Vprime)])
        reg = tf.add_n([tf.reduce_sum(tf.square(self.Uprime), reduction_indices=[0,1]),
                        tf.reduce_sum(tf.square(self.U), reduction_indices=[0,1]),
                        tf.reduce_sum(tf.square(self.V), reduction_indices=[0,1]),
                        tf.reduce_sum(tf.square(self.Vprime), reduction_indices=[0,1])])
        self.loss = reconstruction_loss + (self.lam*reg)

        # Optimizer
        # self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
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
            self.kl_full_iter = 1000

        if 'anneal_kl' in kwargs:
            self.anneal_kl = bool(kwargs['anneal_kl'])
        else:
            self.anneal_kl = True

        super(SVINNMF, self).__init__(*args, **kwargs)

    def _init_vars(self):
        # Input
        self.user_index = tf.placeholder(tf.int32, [None])
        self.item_index = tf.placeholder(tf.int32, [None])
        self.r_target = tf.placeholder(tf.float32, [None])

        # Latents
        self.U_mu = tf.Variable(tf.truncated_normal(
            [self.num_users, self.D], **self.latent_normal_init_params))
        self.U_log_var = tf.Variable(tf.random_uniform(
            [self.num_users, self.D], minval=0.0, maxval=0.5))
        # self.log_U_sigma = tf.log(self.U_sigma+1e-10)

        self.Uprime_mu = tf.Variable(tf.truncated_normal(
            [self.num_users, self.Dprime], **self.latent_normal_init_params))
        self.Uprime_log_var = tf.Variable(tf.random_uniform(
            [self.num_users, self.Dprime], minval=0.0, maxval=0.5))
        # self.log_Uprime_sigma = tf.log(self.Uprime_sigma+1e-10)

        self.V_mu = tf.Variable(tf.truncated_normal(
            [self.num_items, self.D], **self.latent_normal_init_params))
        self.V_log_var = tf.Variable(tf.random_uniform(
            [self.num_items, self.D], minval=0.0, maxval=0.5))
        # self.log_V_sigma = tf.log(self.V_sigma+1e-10)

        self.Vprime_mu = tf.Variable(tf.truncated_normal(
            [self.num_items, self.Dprime], **self.latent_normal_init_params))
        self.Vprime_log_var = tf.Variable(tf.random_uniform(
            [self.num_items, self.Dprime], minval=0.0, maxval=0.5))
        # self.log_Vprime_sigma = tf.log(self.Vprime_sigma+1e-10)

        # Lookups
        self.U_mu_lu = tf.nn.embedding_lookup(self.U_mu, self.user_index)
        self.U_log_var_lu = tf.nn.embedding_lookup(self.U_log_var, self.user_index)

        self.Uprime_mu_lu = tf.nn.embedding_lookup(self.Uprime_mu, self.user_index)
        self.Uprime_log_var_lu = tf.nn.embedding_lookup(self.Uprime_log_var, self.user_index)

        self.V_mu_lu = tf.nn.embedding_lookup(self.V_mu, self.item_index)
        self.V_log_var_lu = tf.nn.embedding_lookup(self.V_log_var, self.item_index)

        self.Vprime_mu_lu = tf.nn.embedding_lookup(self.Vprime_mu, self.item_index)
        self.Vprime_log_var_lu = tf.nn.embedding_lookup(self.Vprime_log_var, self.item_index)

        # priors
        self.p_V = self.p_U = tf.contrib.distributions.MultivariateNormalDiag(mu=tf.zeros(shape=[self.D]),
                                                              diag_stdev=tf.ones([self.D]))

        self.p_Vprime = self.p_Uprime = tf.contrib.distributions.MultivariateNormalDiag(mu=tf.zeros(shape=[self.Dprime]),
                                                                        diag_stdev=tf.ones([self.Dprime]))

        # WEIRD?
        # ------
        # Posterior (q)
        self.q_U = tf.contrib.distributions.MultivariateNormalDiag(mu=self.U_mu_lu, diag_stdev=tf.sqrt(tf.exp(self.U_log_var_lu)))
        self.q_Uprime = tf.contrib.distributions.MultivariateNormalDiag(mu=self.Uprime_mu_lu, diag_stdev=tf.sqrt(tf.exp(self.Uprime_log_var_lu)))
        self.q_V = tf.contrib.distributions.MultivariateNormalDiag(mu=self.V_mu_lu, diag_stdev=tf.sqrt(tf.exp(self.V_log_var_lu)))
        self.q_Vprime = tf.contrib.distributions.MultivariateNormalDiag(mu=self.Vprime_mu_lu, diag_stdev=tf.sqrt(tf.exp(self.Vprime_log_var_lu)))

        # Sample
        # TODO not just mean and more than one sample
        self.U = self.q_U.sample()
        self.Uprime = self.q_Uprime.sample()
        self.V = self.q_V.sample()
        self.Vprime = self.q_Vprime.sample()

        # --------



        # EXPLICIT

        # self.eps_D = tf.random_normal((self.num_data_samples, self.D), 0, 1, dtype=tf.float32)
        # self.eps_Dprime = tf.random_normal((self.num_data_samples, self.Dprime), 0, 1, dtype=tf.float32)
        #
        # self.U = tf.add(self.U_mu_lu,
        #     tf.mul(self.U_sigma_lu, self.eps_D))
        # self.Uprime = tf.add(self.Uprime_mu_lu,
        #     tf.mul(self.Uprime_sigma_lu, self.eps_Dprime))
        # self.V = tf.add(self.V_mu_lu,
        #     tf.mul(self.V_sigma_lu, self.eps_D))
        # self.Vprime = tf.add(self.Vprime_mu_lu,
        #     tf.mul(self.Vprime_sigma_lu, self.eps_Dprime))

        #!!!!!!!!!!!!!!!!!!

        # MLP ("f")
        self.f_input_layer = tf.concat(concat_dim=1,
                                       values=[self.U, self.V, tf.mul(self.Uprime, self.Vprime)])

        self.r_mu, self.mlp_weights = self._build_mlp(self.f_input_layer)
        self.r = tf.squeeze(self.r_mu, squeeze_dims=[1])

        self.rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(self.r, self.r_target))))

        # TODO learn sigma and take it into account in loss
        self.p_r_given_Z = tf.contrib.distributions.Normal(mu=self.r_mu, sigma=math.sqrt(self.r_var)*tf.ones_like(self.r_mu))

        # For KL annealing
        self.kl_weight = tf.placeholder(tf.float32)

    def _get_kl_weight(self, on_iter=100):
        return 1.0/(1 + math.exp(-(25.0/on_iter)*(self._iters-(on_iter/2.0))))

    def _init_ops(self):

        # TODO do i have to tile p_U?
        # self.KL_U = tf.contrib.distributions.kl(self.q_U, self.p_U)
        # self.KL_Uprime = tf.contrib.distributions.kl(self.q_Uprime, self.p_Uprime)
        # self.KL_V = tf.contrib.distributions.kl(self.q_V, self.p_V)
        # self.KL_Vprime = tf.contrib.distributions.kl(self.q_Vprime, self.p_Vprime)
        # self.KL_all = tf.reduce_sum(self.KL_U + self.KL_Uprime + self.KL_V + self.KL_Vprime, reduction_indices=[0])

        # TODO explicit calc
        def KL(mean, log_var, prior_var):
            """Computes KL divergence for a group of univariate normals (ie. every dimension of a latent)."""
            return tf.reduce_sum(tf.log(math.sqrt(prior_var) / tf.sqrt(tf.exp(log_var))) + ((tf.exp(log_var) + tf.square(mean)) / (2.0 * prior_var)), reduction_indices=[0, 1])

        self.KL_U = KL(self.U_mu, self.U_log_var, prior_var=self.U_prior_var)
        self.KL_Uprime = KL(self.Uprime_mu, self.Uprime_log_var, prior_var=self.Uprime_prior_var)
        self.KL_V = KL(self.V_mu, self.V_log_var, prior_var=self.V_prior_var)
        self.KL_Vprime = KL(self.Vprime_mu, self.Vprime_log_var, prior_var=self.Vprime_prior_var)
        self.KL_all = self.KL_U + self.KL_Uprime + self.KL_V + self.KL_Vprime

        # TODO weighting of gradient
        # TODO handle multiple samples
        # self.r_target_stacked = tf.squeeze(tf.reshape(tf.tile(tf.expand_dims(self.r_target, 1), [1, self.num_latent_samples]),
        #                               [self.num_data_samples * self.num_latent_samples, 1]), squeeze_dims=[1])

        # TODO THIS ISN'T WORKING??
        # self.log_prob = tf.reduce_sum(tf.squeeze(self.p_r_given_Z.log_pdf(tf.transpose([self.r_target])), squeeze_dims=[1]), reduction_indices=[0])
        self.log_prob = -(1/(2.0*self.r_var))*tf.reduce_sum(tf.square(tf.sub(self.r_target, self.r)), reduction_indices=[0])

        # self.elbo = self.log_prob-self.KL_all
        self.elbo = self.log_prob-(self.kl_weight*self.KL_all)
        # self.elbo = self.log_prob
        self.loss = -self.elbo

        # TODO REMOVE
        # reconstruction_loss = tf.reduce_sum(tf.square(tf.sub(self.r_target, self.r)), reduction_indices=[0])
        # reg = tf.add_n([tf.sqrt(tf.reduce_sum(tf.square(self.Uprime))), tf.sqrt(tf.reduce_sum(tf.square(self.U))),
        #                 tf.sqrt(tf.reduce_sum(tf.square(self.V))), tf.sqrt(tf.reduce_sum(tf.square(self.Vprime)))])
        # self.loss = reconstruction_loss + (self.lam*reg)
        # ----------

        # squared error loss
        # sqr_loss = tf.square(tf.sub(self.r_target, tf.squeeze(self.r_mu, squeeze_dims=[1])))
        # self.log_prob = -tf.reduce_sum(sqr_loss)
        # self.loss = tf.reduce_sum(sqr_loss)

        # self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
        self.optimizer = tf.train.AdamOptimizer()
        self.optimize_steps = [self.optimizer.minimize(self.loss)]

    def train_iteration(self, data):
        additional_feed = {self.kl_weight: self._get_kl_weight(on_iter=self.kl_full_iter)} if self.anneal_kl else None
        super(SVINNMF, self).train_iteration(data, additional_feed=additional_feed)

    def eval_loss(self, data):
        user_ids = data['user_id']
        item_ids = data['item_id']
        ratings = data['rating']

        feed_dict = {self.user_index: user_ids, self.item_index: item_ids, self.r_target: ratings, self.kl_weight: self._get_kl_weight(on_iter=self.kl_full_iter)}

        # return self.sess.run((self.KL_all, self.kl_weight, -self.log_prob, self.loss), feed_dict=feed_dict)
        return self.sess.run(self.loss, feed_dict=feed_dict)

    def predict(self, user_id, item_id):
        rating = self.sess.run(self.r_mu, feed_dict={self.user_index: [user_id], self.item_index: [item_id]})
        return rating
