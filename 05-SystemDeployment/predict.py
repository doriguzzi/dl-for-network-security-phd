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

import sys
import argparse
import random as rn
import os
import pprint
import warnings

warnings.filterwarnings('ignore')
from traffic_processing import *
from sklearn.metrics import f1_score, accuracy_score,confusion_matrix

# Seed Random Numbers
os.environ['PYTHONHASHSEED']=str(SEED)
np.random.seed(SEED)
rn.seed(SEED)
import tensorflow as tf
config = tf.compat.v1.ConfigProto(inter_op_parallelism_threads=1)

import csv
from tensorflow.keras.models import load_model

import tensorflow.keras.backend as K
tf.random.set_seed(SEED)
K.set_image_data_format('channels_last')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
#config.log_device_placement = True  # to log device placement (on which device the operation ran)

FIELDNAMES = ['Model','Time','Packets', 'Samples', 'DDOS%','Accuracy', 'F1Score','TPR','FPR','TNR','FNR','Source']

def report_results(Y_true, Y_pred, packets, model_name, data_source, prediction_time,writer):
    ddos_rate = '{:04.3f}'.format(sum(Y_pred)/Y_pred.shape[0])

    if Y_true is not None: # if we have the labels, we can compute the classification accuracy
        Y_true = Y_true.reshape((Y_true.shape[0], 1))
        accuracy = accuracy_score(Y_true, Y_pred)

        f1 = f1_score(Y_true, Y_pred)
        tn, fp, fn, tp = confusion_matrix(Y_true, Y_pred,labels=[0,1]).ravel()
        tnr = tn / (tn + fp)
        fpr = fp / (fp + tn)
        fnr = fn / (fn + tp)
        tpr = tp / (tp + fn)

        row = {'Model': model_name, 'Time': '{:04.3f}'.format(prediction_time), 'Packets': packets,
               'Samples': Y_pred.shape[0], 'DDOS%':ddos_rate,'Accuracy':accuracy, 'F1Score':f1,
               'TPR':tpr, 'FPR':fpr, 'TNR':tnr, 'FNR':fnr, 'Source':data_source}
    else:
        row = {'Model': model_name, 'Time': '{:04.3f}'.format(prediction_time), 'Packets': packets,
               'Samples': Y_pred.shape[0], 'DDOS%': ddos_rate, 'Accuracy': "N/A", 'F1Score': "N/A",
               'TPR': "N/A", 'FPR': "N/A", 'TNR': "N/A", 'FNR': "N/A", 'Source': data_source}
    pprint.pprint(row,sort_dicts=False)
    writer.writerow(row)

def main(argv):
    parser = argparse.ArgumentParser(
        description='DDoS attacks detection with Deep Learning',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-p', '--predict', nargs='?', type=str,
                        help='Path to the test set')

    parser.add_argument('-pl', '--predict_live', nargs='?', type=str,
                        help='Network interface of the incoming network traffic')

    parser.add_argument('-pp', '--predict_pcap', nargs='?', type=str,
                        help='Path to the pcap file')

    parser.add_argument('-m', '--model', type=str,
                        help='Path to a pre-trained ANN model')

    parser.add_argument('-o', '--output', type=str, default='./results.csv',
                        help='Output file with the prediction results')

    args = parser.parse_args()

    # do not forget command sudo ./jetson_clocks.sh on the TX2 board before testing
    if args.predict is not None:
        # ADD THE CODE FROM NOTEBOOK "predict-testset" HERE
        # note that args.predict contains the path to a test set
        return

    if args.predict_pcap is not None:
        # ADD THE CODE FROM NOTEBOOK "predict-live" HERE
        # note that args.predict_pcap contains the path to a pcap file
        return

    if args.predict_live is not None:
        # ADD THE CODE FROM NOTEBOOK "predict-live" HERE
        # note that args.predict_live contains the name of a network interface
        return

if __name__ == "__main__":
    main(sys.argv[1:])
