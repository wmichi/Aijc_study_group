import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def importdata(trainfile, testsize, randomseed):
  train_data = pd.read_csv(trainfile)

  X_train, y_train, X_val, y_val = \
   train_test_split(np.array(train_data.iloc[:,1:]),
                    np.array(train_data.iloc[:,0]),
                    test_size=testsize,
                    random_state=randomseed)

  return X_train, y_train, X_val, y_val
