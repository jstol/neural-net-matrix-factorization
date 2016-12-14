#!/usr/bin/env python2.7
from __future__ import absolute_import, print_function
"""Generates predictions."""
# Standard modules
import argparse, json
# Third party modules
import tensorflow as tf
# Package modules
from nnmf.models import NNMF, SVINNMF

if __name__ == '__main__':
    # Set up command line params
    parser = argparse.ArgumentParser(description='Trains/evaluates NNMF models.')
    parser.add_argument('--model', metavar='MODEL_NAME', type=str, choices=['NNMF', 'SVINNMF'],
                        help='the name of the model to use', required=True)
    parser.add_argument('--users', metavar='NUM_USERS', type=int, default=943, # ML 100K has 943 users
                        help='the number of users in the data set')
    parser.add_argument('--movies', metavar='NUM_MOVIES', type=int, default=1682, # ML 100K has 1682 movies
                        help='the number of movies in the data set')
    parser.add_argument('--model-params', metavar='MODEL_PARAMS_JSON', type=str, default='{}',
                        help='JSON string containing model params')
    parser.add_argument('user', metavar='USER_ID', type=int, nargs='?',
                        help='when predicting, the ID of the user to predict a rating for')
    parser.add_argument('item', metavar='ITEM_ID', type=int, nargs='?',
                        help='when predicting, the ID of the item to predict a rating for')

    # Parse args
    args = parser.parse_args()

    model_name = args.model
    model_params = json.loads(args.model_params)
    num_users = args.users
    num_items = args.movies
    user_id = args.user
    item_id = args.item

    print('Building network & initializing variables')
    if model_name == 'NNMF':
        model = NNMF(num_users, num_items, **model_params)
    elif model_name == 'SVINNMF':
        model = SVINNMF(num_users, num_items, **model_params)
    else:
        raise NotImplementedError("Model '{}' not implemented".format(model_name))

    with tf.Session() as sess:
        model.init_sess(sess)
        saver = tf.train.Saver()

        print('Loading model')
        saver.restore(sess, model.model_filename)
        user_id, item_id = args.user, args.item
        rating = model.predict(user_id - 1, item_id - 1)

    print("Predicted rating for user '{}' & item '{}': {}".format(user_id, item_id, rating))
