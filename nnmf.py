#!/usr/bin/env python2.7
from __future__ import absolute_import, print_function
"""NNMF model."""
# Standard modules
import sys
from math import sqrt
# Third party modules
import tensorflow as tf
import pandas as pd
import numpy as np

# Various constants
model_filename = 'model/model.ckpt'
early_stop_max_iter = 5
max_iters = 1000
batch_size = None
num_users = 943
num_items = 1682
# Model/loss params
lam = 0.01
learning_rate = 0.001
D = 10
Dprime = 60
# num_hidden = 3
hidden_units_per_layer = 50
latent_normal_init_params = {
    'mean': 0.0,
    'stddev': 0.1,
}

def weight_init_range(n_in, n_out):
    range = 4.0*sqrt(6.0)/sqrt(n_in + n_out)
    return {
        'minval': -range,
        'maxval': range,
    }

if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print('train or predict?')
        sys.exit(0)

    # Define computation graph
    print('Building network')

    # Input
    user_index = tf.placeholder(tf.int32, [None])
    item_index = tf.placeholder(tf.int32, [None])
    r_target = tf.placeholder(tf.float32, [None])

    # Latents
    U = tf.Variable(tf.random_normal([num_users, D], **latent_normal_init_params))
    Uprime = tf.Variable(tf.random_normal([num_users, Dprime], **latent_normal_init_params))
    V = tf.Variable(tf.random_normal([num_items, D], **latent_normal_init_params))
    Vprime = tf.Variable(tf.random_normal([num_items, Dprime], **latent_normal_init_params))

    # Lookups
    U_lu = tf.nn.embedding_lookup(U, user_index)
    Uprime_lu = tf.nn.embedding_lookup(Uprime, user_index)
    V_lu = tf.nn.embedding_lookup(V, item_index)
    Vprime_lu = tf.nn.embedding_lookup(Vprime, item_index)

    # MLP ("f") - TODO make this nicer, if possible (loop?)
    f_input_layer = tf.concat(concat_dim=1, values=[U_lu, V_lu, tf.mul(Uprime_lu, Vprime_lu)])
    num_f_inputs = f_input_layer.get_shape().as_list()[1]

    # MLP weights picked uniformly from +/- 4*sqrt(6)/sqrt(n_in + n_out)
    weights = {
        'h1': tf.Variable(tf.random_uniform([num_f_inputs, hidden_units_per_layer], **weight_init_range(num_f_inputs, hidden_units_per_layer))),
        'h2': tf.Variable(tf.random_uniform([hidden_units_per_layer, hidden_units_per_layer], **weight_init_range(hidden_units_per_layer, hidden_units_per_layer))),
        'h3': tf.Variable(tf.random_uniform([hidden_units_per_layer, hidden_units_per_layer], **weight_init_range(hidden_units_per_layer, hidden_units_per_layer))),
        'out': tf.Variable(tf.random_uniform([hidden_units_per_layer, 1], **weight_init_range(hidden_units_per_layer, 1))),
    }
    # MLP layers
    layer_1 = tf.nn.sigmoid(tf.matmul(f_input_layer, weights['h1']))
    layer_2 = tf.nn.sigmoid(tf.matmul(layer_1, weights['h2']))
    layer_3 = tf.nn.sigmoid(tf.matmul(layer_2, weights['h3']))
    r = tf.squeeze(tf.matmul(layer_3, weights['out']), squeeze_dims=[1])

    # Define loss
    def l2(t):
        return tf.nn.l2_loss(t)

    loss = tf.add(tf.reduce_sum(tf.square(tf.sub(r_target, r))), tf.scalar_mul(lam, tf.add_n([l2(Uprime), l2(U), l2(V), l2(Vprime)])))

    # Set up optimizers
    opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    f_train_step = opt.minimize(loss, var_list=weights.values())
    latent_train_step = opt.minimize(loss, var_list=[U, Uprime, V, Vprime])

    saver = tf.train.Saver()
    with tf.Session() as sess:
        # Train
        if sys.argv[1] == 'train':
            # Read in data
            print('Reading in data')

            data = pd.read_csv('data/ml-100k/u.data', delimiter='\t', header=None,
                               names=['user_id', 'item_id', 'rating', 'timestamp'])
            data.drop('timestamp', axis=1, inplace=True)
            # Make user/item indices start at 0 (in MovieLens data set they start at 1)
            data['user_id'] = data['user_id'] - 1
            data['item_id'] = data['item_id'] - 1

            # Shuffle
            data = (data.iloc[np.random.permutation(len(data))]).reset_index(drop=True)

            # Split data set into (all) train/test
            all_train_ratio = 0.98
            num_all_train = int(len(data)*all_train_ratio)
            num_test = len(data)-num_all_train
            all_train_data = data.head(num_all_train)
            test_data = data.tail(num_test)

            # Split up (all) train data into train/validation
            train_ratio = 0.9
            num_train = int(len(all_train_data)*train_ratio)
            num_valid = len(all_train_data)-num_train
            train_data = all_train_data.head(num_train)
            valid_data = all_train_data.tail(num_valid)

            print("Data subsets:")
            print("Train: {}".format(len(train_data)))
            print("Validation: {}".format(len(valid_data)))
            print("Test: {}".format(len(test_data)))

            # Set up validation data
            valid_user_ids = valid_data['user_id']
            valid_item_ids = valid_data['item_id']
            valid_ratings = valid_data['rating']
            valid_feed_dict = {user_index: valid_user_ids, item_index: valid_item_ids, r_target: valid_ratings}

            # Set up test data
            test_user_ids = test_data['user_id']
            test_item_ids = test_data['item_id']
            test_ratings = test_data['rating']
            test_feed_dict = {user_index: test_user_ids, item_index: test_item_ids, r_target: test_ratings}

            print('Initializing session')
            init = tf.initialize_all_variables()
            sess.run(init)

            # Define RMSE so we can keep track of it as training progresses
            rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(r, r_target))))

            # Print initial values
            batch = train_data.sample(batch_size) if batch_size else train_data
            user_ids = batch['user_id']
            item_ids = batch['item_id']
            ratings = batch['rating']
            feed_dict = {user_index: user_ids, item_index: item_ids, r_target: ratings}
            train_error = sess.run(rmse, feed_dict=feed_dict)
            valid_error = sess.run(rmse, feed_dict=valid_feed_dict)
            print(train_error, valid_error)

            # Optimize
            prev_valid_error = float("Inf")
            early_stop_iters = 0
            for i in xrange(max_iters):
                batch = train_data.sample(batch_size) if batch_size else train_data

                user_ids = batch['user_id']
                item_ids = batch['item_id']
                ratings = batch['rating']
                feed_dict = {user_index: user_ids, item_index: item_ids, r_target: ratings}
                # Optimize weights
                sess.run(f_train_step, feed_dict=feed_dict)
                # Optimize latents
                sess.run(latent_train_step, feed_dict=feed_dict)
                # Evaluate
                train_error = sess.run(rmse, feed_dict=feed_dict)
                valid_error = sess.run(rmse, feed_dict=valid_feed_dict)
                print(train_error, valid_error)

                # Checkpointing/early stopping
                early_stop_iters += 1
                if valid_error < prev_valid_error:
                    prev_valid_error = valid_error
                    early_stop_iters = 0
                    saver.save(sess, model_filename)
                elif early_stop_iters == early_stop_max_iter:
                    print("Early stopping ({} vs. {})...".format(prev_valid_error, valid_error))
                    break

            print('Loading best checkpointed model')
            saver.restore(sess, model_filename)
            test_error = sess.run(rmse, feed_dict=test_feed_dict)
            print("Final test RMSE: {}".format(test_error))

        elif sys.argv[1] == 'predict':
            print('Loading model')
            saver.restore(sess, model_filename)
            user_id = int(sys.argv[2])
            item_id = int(sys.argv[3])
            rating = sess.run(r, feed_dict={user_index: [user_id-1], item_index: [item_id-1]})
            print("Predicted rating for user '{}' & item '{}': {}".format(user_id, item_id, rating[0]))
