import numpy as np
import pandas as pd

from sklearn import tree
import pydotplus

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from sklearn.externals import joblib

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

# Read Train Data
X = pd.read_csv('./data/all_train.csv')
y = pd.read_csv('./data/all_label.csv')

rf0 = GradientBoostingRegressor(n_estimators=2000, learning_rate=0.001, loss='ls')
rf0.fit(X, y)
y_pred = rf0.predict(X)

print('R2 score (Train): %f' % r2_score(y, y_pred))

joblib.dump(rf0, './model/sklearn_GBDT_model_1500.model')
