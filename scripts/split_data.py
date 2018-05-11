#!/usr/bin/env python2.7
from __future__ import absolute_import, print_function
"""Simple utility to separate data into training, test, and validation."""
# Standard libraries
import argparse, os
# Third party modules
import pandas as pd
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simple utility to separate data into training, test, and validation.')
    parser.add_argument('--input', metavar='INPUT_FILE', type=str, default='data/ml-100k/u.data',
                        help='the location of the input file')
    parser.add_argument('--outfolder', metavar='OUTPUT_FOLDER', type=str, default='data/ml-100k/split',
                        help='the location of the folder to output to')
    parser.add_argument('--outprefix', metavar='OUTPUT_PREFIX', type=str, default='u.data',
                        help='string to append to front of output filenames')
    parser.add_argument('--delimiter', metavar='DELIMITER', type=str, default='\t',
                        help='delimiter to use when parsing/writing files')
    parser.add_argument('--train-test-r', metavar='TRAIN_TEST_RATIO', type=float, default=0.9,
                        help='Ratio of training data to test data')
    parser.add_argument('--train-valid-r', metavar='TRAIN_VALID_RATIO', type=float, default=0.98,
                        help='Ratio of training data to validation data')

    args = parser.parse_args()
    input_filename = args.input
    output_folder = args.outfolder
    output_prefix = args.outprefix
    delimiter = args.delimiter
    train_test_r = args.train_test_r
    train_valid_r = args.train_valid_r

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    # Read in data
    print('Reading in data')

    data = pd.read_csv(input_filename, delimiter=delimiter, header=None,
                       names=['user_id', 'item_id', 'rating', 'timestamp'])
    data.drop('timestamp', axis=1, inplace=True)

    # Shuffle
    data = (data.iloc[np.random.permutation(len(data))]).reset_index(drop=True)

    # Split data set into (all) train/test
    all_train_ratio = train_test_r
    num_all_train = int(len(data)*all_train_ratio)
    num_test = len(data)-num_all_train
    all_train_data = data.head(num_all_train)
    test_data = (data.tail(num_test)).reset_index(drop=True)

    # Split up (all) train data into train/validation
    train_ratio = train_valid_r
    num_train = int(len(all_train_data)*train_ratio)
    num_valid = len(all_train_data)-num_train
    train_data = (all_train_data.head(num_train)).reset_index(drop=True)
    valid_data = (all_train_data.tail(num_valid)).reset_index(drop=True)

    print("Data subsets:")
    print("Train: {}".format(len(train_data)))
    print("Validation: {}".format(len(valid_data)))
    print("Test: {}".format(len(test_data)))

    # Write data to file
    common = {'header': False, 'sep': delimiter, 'index': False}
    train_data.to_csv(os.path.join(output_folder, "{}.train".format(output_prefix)), **common)
    valid_data.to_csv(os.path.join(output_folder, "{}.valid".format(output_prefix)), **common)
    test_data.to_csv(os.path.join(output_folder, "{}.test".format(output_prefix)), **common)
