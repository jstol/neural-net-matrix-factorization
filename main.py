#!/usr/bin/env python2.7
from __future__ import absolute_import, print_function
"""Trains NNMF models and generates predictions."""
# Standard modules
import argparse, json
# Third party modules
import tensorflow as tf
import pandas as pd
# Package modules
from nnmf.models import NNMF, SVINNMF

if __name__ == '__main__':
    # Set up command line params
    parser = argparse.ArgumentParser(description='Trains NNMF models and generates predictions.')
    parser.add_argument('--model', metavar='MODEL_NAME', type=str, choices=['NNMF', 'SVINNMF'],
                        help='the name of the model to use')
    parser.add_argument('--model-params', metavar='MODEL_PARAMS_JSON', type=str, required=False,
                        help='JSON string containing model params')
    parser.add_argument('--mode', metavar='MODE', type=str, choices=['train', 'test', 'predict'],
                        help='the mode to run the program in')
    parser.add_argument('--train', metavar='TRAIN_INPUT_FILE', type=str, default='data/ml-100k/split/u.data.train',
                        help='the location of the training set\'s input file')
    parser.add_argument('--valid', metavar='VALID_INPUT_FILE', type=str, default='data/ml-100k/split/u.data.valid',
                        help='the location of the validation set\'s input file')
    parser.add_argument('--test', metavar='TEST_INPUT_FILE', type=str, default='data/ml-100k/split/u.data.test',
                        help='the location of the test set\'s input file')
    parser.add_argument('--delim', metavar='DELIMITER', type=str, default='\t',
                        help='the delimiter to use when parsing input files')
    parser.add_argument('--cols', metavar='COL_NAMES', type=str, default=['user_id', 'item_id', 'rating'],
                        help='the column names of the input data', nargs='+')
    parser.add_argument('--users', metavar='NUM_USERS', type=int, default=943, # ML 100K has 943 users
                        help='the number of users in the data set')
    parser.add_argument('--movies', metavar='NUM_MOVIES', type=int, default=1682, # ML 100K has 1682 movies
                        help='the number of movies in the data set')
    parser.add_argument('--batch', metavar='BATCH_SIZE', type=int, default=25000,
                        help='the batch size to use when doing gradient descent')
    parser.add_argument('--no-early', default=True, action='store_false',
                        help='disable early stopping')
    parser.add_argument('--early-stop-max-iter', metavar='EARLY_STOP_MAX_ITER', type=int, default=250,
                        help='the maximum number of iterations to let the model continue training after reaching a '
                             'minimum validation error')
    parser.add_argument('--max-iters', metavar='MAX_ITERS', type=int, default=10000,
                        help='the maximum number of iterations to allow the model to train for')
    parser.add_argument('user', metavar='USER_ID', type=int, nargs='?',
                        help='when predicting, the ID of the user to predict a rating for')
    parser.add_argument('item', metavar='ITEM_ID', type=int, nargs='?',
                        help='when predicting, the ID of the item to predict a rating for')

    # Parse args
    args = parser.parse_args()
    # Global args
    model_name = args.model
    model_params = json.loads(args.model_params) if args.model_params else {}
    num_users = args.users
    num_items = args.movies
    mode = args.mode

    with tf.Session() as sess:
        # Define computation graph & Initialize
        print('Building network & initializing variables')
        if model_name == 'NNMF':
            model = NNMF(num_users, num_items, **model_params)
        elif model_name == 'SVINNMF':
            model = SVINNMF(num_users, num_items, **model_params)

        model.init_sess(sess)
        saver = tf.train.Saver()

        # Train
        if mode in ('train', 'test'):
            # Read in train/test specific args
            train_filename = args.train
            valid_filename = args.valid
            test_filename = args.test
            delimiter = args.delim
            col_names = args.cols
            batch_size = args.batch
            use_early_stop = not(args.no_early)
            early_stop_max_iter = args.early_stop_max_iter
            max_iters = args.max_iters

            # Process data
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

            if mode == 'train':
                # Print initial values
                batch = train_data.sample(batch_size) if batch_size else train_data
                train_error = model.eval_loss(batch)
                train_rmse = model.eval_rmse(batch)
                valid_error = model.eval_loss(valid_data)
                valid_rmse = model.eval_rmse(valid_data)
                print("{} {:2f} | {} {:2f}".format(train_error, train_rmse, valid_error, valid_rmse))

                # Optimize
                prev_valid_error = float("Inf")
                early_stop_iters = 0
                for i in xrange(max_iters):
                    # Run SGD
                    batch = train_data.sample(batch_size) if batch_size else train_data
                    model.train_iteration(batch)

                    # Evaluate
                    train_error = model.eval_loss(batch)
                    train_rmse = model.eval_rmse(batch)
                    valid_error = model.eval_loss(valid_data)
                    valid_rmse = model.eval_rmse(valid_data)
                    print("{} {:2f} | {} {:2f}".format(train_error, train_rmse, valid_error, valid_rmse))

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

        elif mode == 'predict':
            print('Loading model')
            saver.restore(sess, model.model_filename)
            user_id, item_id = args.user, args.item
            rating = model.predict(user_id-1, item_id-1)
            print("Predicted rating for user '{}' & item '{}': {}".format(user_id, item_id, rating))
