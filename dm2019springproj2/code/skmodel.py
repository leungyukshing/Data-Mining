import numpy as np
from mpi4py import MPI
import pandas as pd

from sklearn import tree

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# Read Train Data
X = pd.read_csv('../data/train1.csv')
y = pd.read_csv('../data/label1.csv')

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Construct a Regression Tree
regressionTree = tree.DecisionTreeRegressor(max_depth=5, min_samples_split=10, max_features='sqrt')
regressionTree = regressionTree.fit(X_train, y_train)
    
# validation
y_test_prediction = regressionTree.predict(X_test)
mse_error = mean_squared_error(y_test, y_test_prediction)

print("MSE_error is " + str(mse_error))