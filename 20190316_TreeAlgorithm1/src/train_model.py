import os
import sys
import json
import argparse
from time import strftime, gmtime
import numpy as np
import pandas as pd
from ImportData import *
from models import train_model

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--modelname', default='DecisionTree',help='classification algorithm')
  parser.add_argument('--trainfile', default='../data/basedata/train.csv',help='train data path.')
  parser.add_argument('--testfile', default='../data/basedata/test.csv',help='test data path.')
  parser.add_argument('--testsize', type=int, default=0.2 ,help='split test size')
  parser.add_argument('--randomseed', type=int, default=0, help='random seed')
  parser.add_argument('--n_estimators', type=int, default=10, help='number of estimators used for RandomForest or GradientBoosting')
  parser.add_argument('--oob_score', type=bool, default=False, help='Whethre to use out-of-bag samples to estimate the generalization accuracy')

  return parser.parse_args()

if __name__=='__main__':
  execute_time = strftime("%Y%m%d_%H%M%S", gmtime())
  args = parse_args()

  # logging params
  param_filename = execute_time + '_TreeClassifier_params.json'
  param_filepath = os.path.join('../log/args_json', param_filename)
  with open(param_filepath, 'w') as f:
    json.dump(vars(args), f)

  X_train, X_val, y_train, y_val = importdata(args.trainfile, args.testsize, args.randomseed)

  model, train_acc, val_acc = train_model(X_train, X_val, y_train, y_val, args)

  print('{0}(Accuracy): \n Train: {1:.3f} \n Validation:{2:.3f}'.format(args.modelname, train_acc, val_acc))
