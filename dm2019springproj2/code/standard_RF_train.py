import numpy as np
from mpi4py import MPI
import pandas as pd

from sklearn import tree
import pydotplus

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from sklearn.externals import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Read Train Data
X = pd.read_csv('../data/all_train.csv')
y = pd.read_csv('../data/all_label.csv')

rf0 = RandomForestRegressor(oob_score=True, random_state=10)
rf0.fit(X, y)
y_pred = rf0.predict(X)

print(rf0.oob_score_)
print('R2 score (Train): %f' % r2_score(y, y_pred))

joblib.dump(rf0, '../model/sklearn_model.model')
