import os
import sys
from time import strftime, gmtime
from shutil import  copyfile
import numpy as np
import pandas as pd

def create_dataset(filename, filepath, train=True):
  # data import and select columns
  data = pd.read_csv(filename)
  if train:
    data = data.loc[:,['Survived','Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare' ,'Embarked']]
  else:
    data = data.loc[:,['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare' ,'Embarked']]

  # one-hot encoding and fill nan
  columns = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']
  data = pd.get_dummies(data, columns=columns, dummy_na=True)
  data.Age = data.Age.fillna(data.Age.mean())

  # export DataFrame as csv format file
  if train:
    csv_filename = os.path.join(filepath, 'train.csv')
  else:
    csv_filename = os.path.join(filepath, 'test.csv')
  data.to_csv(csv_filename, index=None)


if __name__=='__main__':
  execute_time = strftime("%Y%m%d_%H%M%S", gmtime())
  if not os.path.exists('../data/basedata'):
    filepath = '../data/basedata'
    os.mkdir(filepath)
  else:
    filepath = os.path.join('../data', execute_time)
    os.mkdir(filepath)

  train_filename = '../data/org/train.csv'
  create_dataset(train_filename, filepath)

  test_filename = '../data/org/test.csv'
  create_dataset(test_filename, filepath, train=False)

  # preserve script file to check the process
  script_filename = os.path.join(filepath, 'make_script.py')
  copyfile(os.path.abspath(sys.argv[0]), script_filename)
