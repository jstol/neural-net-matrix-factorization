#!/usr/bin/env python2.7
from __future__ import absolute_import, print_function
"""NNMF model."""
# Standard modules
import sys
# Third party modules
import tensorflow as tf
import pandas as pd
import numpy as np
# Package modules
from models import NNMF, SVINNMF

# Various constants
early_stop_max_iter = 50
max_iters = 10000
batch_size = None
num_users = 943
num_items = 1682

if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print('train or predict?')
        sys.exit(0)

    with tf.Session() as sess:
        # Define computation graph & Initialize
        print('Building network & initializing variables')
        # model = NNMF(num_users, num_items)
        model = SVINNMF(num_users, num_items)
        model.init_sess(sess)
        saver = tf.train.Saver()

        # Train
        if sys.argv[1] == 'train':
            # Read in data
            print('Reading in data')

            data = pd.read_csv('data/ml-100k/u.data', delimiter='\t', header=None,
                               names=['user_id', 'item_id', 'rating', 'timestamp'])
            data.drop('timestamp', axis=1, inplace=True)
            # Make user/item indices start at 0 (in MovieLens data set they start at 1)
            data['user_id'] = data['user_id']-1
            data['item_id'] = data['item_id']-1

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

            # Print initial values
            batch = train_data.sample(batch_size) if batch_size else train_data
            train_error = model.eval_rmse(batch)
            valid_error = model.eval_rmse(valid_data)
            print("{:2f} {:2f}".format(train_error, valid_error))

            # Optimize
            prev_valid_error = float("Inf")
            early_stop_iters = 0
            for i in xrange(max_iters):
                # Run (S)GD
                batch = train_data.sample(batch_size) if batch_size else train_data
                model.train_iteration(batch)

                # Evaluate
                train_error = model.eval_rmse(batch)
                valid_error = model.eval_rmse(valid_data)
                print("{:2f} {:2f}".format(train_error, valid_error))

                # Checkpointing/early stopping
                early_stop_iters += 1
                if valid_error < prev_valid_error:
                    prev_valid_error = valid_error
                    early_stop_iters = 0
                    saver.save(sess, model.model_filename)
                elif early_stop_iters == early_stop_max_iter:
                    print("Early stopping ({} vs. {})...".format(prev_valid_error, valid_error))
                    break

            print('Loading best checkpointed model')
            saver.restore(sess, model.model_filename)
            test_error = model.eval_rmse(test_data)
            print("Final test RMSE: {}".format(test_error))

        elif sys.argv[1] == 'predict':
            print('Loading model')
            saver.restore(sess, model.model_filename)
            user_id = int(sys.argv[2])
            item_id = int(sys.argv[3])
            rating = model.predict(user_id-1, item_id-1)
            print("Predicted rating for user '{}' & item '{}': {}".format(user_id, item_id, rating))
