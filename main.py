#!/usr/bin/env python2.7
from __future__ import absolute_import, print_function
"""NNMF model."""
# Standard modules
import sys, logging
# Third party modules
import tensorflow as tf
import pandas as pd
import numpy as np
# Package modules
from models import NNMF, SVINNMF

# Various constants
use_early_stop = True
early_stop_max_iter = 2500
max_iters = 10000
batch_size = 25000
# batch_size = None
num_users = 943
num_items = 1682

train_filename = 'data/ml-100k/split/u.data.train'
valid_filename = 'data/ml-100k/split/u.data.valid'
test_filename = 'data/ml-100k/split/u.data.test'
delimiter = '\t'
col_names = ['user_id', 'item_id', 'rating']

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print('train or predict?')
        sys.exit(0)

    with tf.Session() as sess:
        # Define computation graph & Initialize
        print('Building network & initializing variables')
        # model = NNMF(num_users, num_items, lam=0.5, model_filename='model/nnmf_RUNNING-A-TESTTTTTTTTTTTTT-STANDARD.ckpt')
        model = SVINNMF(num_users, num_items, kl_full_iter=max_iters/3, model_filename='model/svi_nnmf_STOCHASTIC-WITH_KL-TESTTTTTTTT3.ckpt')
        model.init_sess(sess)
        saver = tf.train.Saver()

        # Train
        if sys.argv[1] in ('train', 'test'):
            print("Reading in data")
            train_data = pd.read_csv(train_filename, delimiter=delimiter, header=None, names=col_names)
            train_data['user_id'] = train_data['user_id'] - 1
            train_data['item_id'] = train_data['item_id'] - 1
            valid_data = pd.read_csv(valid_filename, delimiter=delimiter, header=None, names=col_names)
            valid_data['user_id'] = valid_data['user_id'] - 1
            valid_data['item_id'] = valid_data['item_id'] - 1
            test_data = pd.read_csv(test_filename, delimiter=delimiter, header=None, names=col_names)
            test_data['user_id'] = test_data['user_id'] - 1
            test_data['item_id'] = test_data['item_id'] - 1

            if sys.argv[1] == 'train':
                # Print initial values
                batch = train_data.sample(batch_size) if batch_size else train_data
                train_error = model.eval_loss(batch)
                train_rmse = model.eval_rmse(batch)
                valid_error = model.eval_loss(valid_data)
                valid_rmse = model.eval_rmse(valid_data)
                print("{} {:2f} | {} {:2f}".format(train_error, train_rmse, valid_error, valid_rmse))

                # TODO PUT BACK IF NEEDED
                # batch = train_data.sample(batch_size) if batch_size else train_data
                # train_error = model.eval_loss(batch)
                # print(train_error)

                # Optimize
                prev_valid_error = float("Inf")
                early_stop_iters = 0
                for i in xrange(max_iters):
                    # Run (S)GD
                    batch = train_data.sample(batch_size) if batch_size else train_data
                    model.train_iteration(batch)

                    # Evaluate
                    train_error = model.eval_loss(batch)
                    train_rmse = model.eval_rmse(batch)
                    valid_error = model.eval_loss(valid_data)
                    valid_rmse = model.eval_rmse(valid_data)
                    print("{} {:2f} | {} {:2f}".format(train_error, train_rmse, valid_error, valid_rmse))

                    # TODO put back IF NEEDED
                    # train_error = model.eval_loss(batch)
                    # print(train_error)

                    # Checkpointing/early stopping
                    if use_early_stop:
                        early_stop_iters += 1
                        if valid_error < prev_valid_error:
                            prev_valid_error = valid_error
                            early_stop_iters = 0
                            saver.save(sess, model.model_filename)
                        elif early_stop_iters == early_stop_max_iter:
                            print("Early stopping ({} vs. {})...".format(prev_valid_error, valid_error))
                            break
                    else:
                        saver.save(sess, model.model_filename)

            print('Loading best checkpointed model')
            saver.restore(sess, model.model_filename)
            train_error = model.eval_rmse(train_data)
            print("Final train RMSE: {}".format(train_error))
            test_error = model.eval_rmse(test_data)
            print("Final test RMSE: {}".format(test_error))

        elif sys.argv[1] == 'predict':
            print('Loading model')
            saver.restore(sess, model.model_filename)
            user_id = int(sys.argv[2])
            item_id = int(sys.argv[3])
            rating = model.predict(user_id-1, item_id-1)
            print("Predicted rating for user '{}' & item '{}': {}".format(user_id, item_id, rating))
