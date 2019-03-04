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
  parser.add_argument('--modelname', default='DecisionTreeClassifier',help='classification algorithm')
  parser.add_argument('--trainfile', default='../data/basedata/train.csv',help='train data path.')
  parser.add_argument('--testfile', default='../data/basedata/test.csv',help='train data path.')
  parser.add_argument('--testsize', type=int, default=0.2 ,help='split test size')
  parser.add_argument('--randomseed', type=int, default=0, help='random seed for train_test_split')

  return parser.parse_args()

if __name__=='__main__':
  execute_time = strftime("%Y%m%d_%H%M%S", gmtime())
  args = parse_args()

  # logging params
  param_filename = execute_time + '_TreeClassifier_params.json'
  param_filepath = os.path.join('../log/args_json', param_filename)
  print(args)
  with open(param_filepath, 'w') as f:
    json.dump(vars(args), f)

  X_train, X_val, y_train, y_val = importdata(args.trainfile, args.testsize, args.randomseed)

  model, train_acc, val_acc = train_model(args.modelname, X_train, X_val, y_train, y_val)
  print(train_acc)
  print(val_acc)
