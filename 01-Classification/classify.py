# Author: Roberto Doriguzzi-Corin
# Project: Lecture on Intrusion Detection with Deep Learning
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Usage: python classify.py -d ../Datasets/DOS2019/ ../Datasets/IDS2017/ -m ./Models/

import numpy as np
import argparse
import h5py
import glob
import time
import sys
import csv
import os
import logging
from tensorflow.keras.models import load_model
from sklearn.metrics import  f1_score, accuracy_score
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)

def load_dataset(path):
    filename = glob.glob(path)[0]
    dataset = h5py.File(filename, "r")
    set_x_orig = np.array(dataset["set_x"][:])  # features
    set_y_orig = np.array(dataset["set_y"][:])  # labels

    X_train = np.reshape(set_x_orig, (set_x_orig.shape[0], set_x_orig.shape[1], set_x_orig.shape[2], 1))
    Y_train = set_y_orig

    return X_train, Y_train

def compute_metrics(Y_true, Y_pred):
    Y_true = Y_true.reshape((Y_true.shape[0], 1))
    accuracy = accuracy_score(Y_true, Y_pred)
    f1 = f1_score(Y_true, Y_pred)
    return accuracy, f1

def main(argv):
    help_string = 'Usage: python3 classify.py --dataset_folder <dataset_folder> --model_folder <model_folder>'

    parser = argparse.ArgumentParser(
        description='Test classification with a pre-trained model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-d', '--dataset_folder', nargs='+', type=str,
                        help='Folder(s) with the dataset')

    parser.add_argument('-m', '--model_folder', nargs='?', type=str, default="./log",
                        help='Folder with a pre-trained model saved in h5 format')

    args = parser.parse_args()

    if args.dataset_folder is not None and args.model_folder is not None:
        model_list = glob.glob(args.model_folder + "/*.h5") # list of pre-trained models
        dataset_filelist = []
        for folder in args.dataset_folder:
            dataset_filelist += glob.glob(folder + "/*test.hdf5") # list of test sets

        classify_fieldnames = ['Dataset', 'Samples', 'Time', 'Accuracy', 'F1Score', 'Model']
        predict_file = open('./results.csv', 'a', newline='')
        predict_file.truncate(0)  # clean the file content (as we open the file in append mode)
        predict_writer = csv.DictWriter(predict_file, fieldnames=classify_fieldnames)
        predict_writer.writeheader()
        predict_file.flush()

        for model_path in model_list:
            model = load_model(model_path)
            model_filename = model_path.split('/')[-1].split('.')[0]
            for dataset_file in dataset_filelist:
                dataset_filename = dataset_file.split('/')[-1].split('.')[0]
                X, Y = load_dataset(dataset_file)
                Y_true = Y
                pt0 = time.time()
                Y_pred = np.squeeze(model.predict(X, batch_size=2048) > 0.5)
                pt1 = time.time()
                accuracy, f1 = compute_metrics(Y_true,Y_pred)
                row = {'Dataset': dataset_filename, 'Samples': Y_true.shape[0], 'Time': '{:10.3f}'.format(pt1-pt0), 'Accuracy': accuracy,
                       'F1Score': f1, 'Model': model_filename}
                print (row)
                predict_writer.writerow(row)

        predict_file.close()
        print("Classification results saved in file: ", predict_file.name)

if __name__ == "__main__":
    main(sys.argv[1:])