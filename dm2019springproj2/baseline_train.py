import numpy as np
from mpi4py import MPI
import pandas as pd

from sklearn import tree
import pydotplus

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler


# Get MPI info
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if (size != 5):
    print("Error!")
    exit()

def single_regression_tree(X, y, params):
    '''
    Args:
        X: 训练集输入数据
        y: 训练集标签值
        params: 回归树参数[max_depth, min_samples_split, max_features]
    '''
    # split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # print(type(X_train))
    # print(X_train.shape[0])

    # multiple Regression Tree
    '''
    X_train_arr = []
    y_train_arr = []
    for i in range(0, 100):
        X_train_temp = X_train[i*14000: (i+1) * 14000]
        y_train_temp = y_train[i*14000: (i+1) * 14000]
        X_train_arr.append(X_train_temp)
        y_train_arr.append(y_train_temp)
    '''
    X_train_arr = []
    y_train_arr = []
    # Tree number = 500
    for i in range(0, 150):
        # Randomly choose 5000 samples to build tree
        X_train_temp = pd.DataFrame()
        y_train_temp = pd.DataFrame()
        for j in range(0, 20000):
            k = np.random.randint(1, X_train.shape[0])
            X_train_t = X_train.iloc[k-1:k, :]
            y_train_t = y_train.iloc[k-1:k, :]
            X_train_temp =  X_train_temp.append(X_train_t)
            y_train_temp = y_train_temp.append(y_train_t)
        #X_train_temp = X_train[i*14000: (i+1) * 14000]
        #y_train_temp = y_train[i*14000: (i+1) * 14000]
        #print(len(X_train_temp))
        #print(len(y_train_temp))
        X_train_arr.append(X_train_temp)
        y_train_arr.append(y_train_temp)
        

    # multiple train
    regressionTreeArr = []
    mse_error_testArr = []
    r2score_testArr = []

    mse_error_trainArr = []
    r2score_trainArr = []

    for i in range(0,150):
        # train
        regressionTree = tree.DecisionTreeRegressor(max_depth=None, min_samples_split=15, min_samples_leaf=23, max_features=10, criterion='mse', splitter='random')
        regressionTree = tree.DecisionTreeRegressor()
        #regressionTree = tree.DecisionTreeRegressor()
        regressionTree = regressionTree.fit(X_train_arr[i], y_train_arr[i])

        # train loss
        y_train_prediction_temp = regressionTree.predict(X_train_arr[i])
        mse_error_train_temp = mean_squared_error(y_train_arr[i], y_train_prediction_temp)
        r2score_train_temp = r2_score(y_train_arr[i], y_train_prediction_temp)

        # save
        mse_error_trainArr.append(mse_error_train_temp)
        r2score_trainArr.append(r2score_train_temp)

        # validation
        y_test_prediction_temp = regressionTree.predict(X_test)
        mse_error_test_temp = mean_squared_error(y_test, y_test_prediction_temp)
        r2score_test_temp = r2_score(y_test, y_test_prediction_temp)

        # save
        regressionTreeArr.append(regressionTree)
        mse_error_testArr.append(mse_error_test_temp)
        r2score_testArr.append(r2score_test_temp)

    # Construct a Regression Tree
    '''
    regressionTree = tree.DecisionTreeRegressor(max_depth=params[0], min_samples_split=params[1], max_features=params[2], criterion='mse')
    regressionTree = regressionTree.fit(X_train, y_train)
    '''

    # validation
    '''
    y_test_prediction = regressionTree.predict(X_test)
    mse_error = mean_squared_error(y_test, y_test_prediction)
    r2score = r2_score(y_test, y_test_prediction)
    # train model use all data
    regressionTree = tree.DecisionTreeRegressor(max_depth=None, min_samples_split=params[1], max_features=None)
    regressionTree = regressionTree.fit(X, y)
    '''
    # Return both the model and the validation error
    # return regressionTree, mse_error, r2score
    return regressionTreeArr, mse_error_testArr, r2score_testArr, mse_error_trainArr, r2score_trainArr
    
# print info
print("Hi, I am " + str(rank + 1) + " process")

# Read Train Data
X = pd.read_csv('./data/train' + str(rank + 1) + '.csv')
y = pd.read_csv('./data/label' + str(rank + 1) + '.csv')

# Data Normalization(optional)
'''
sc = StandardScaler()
sc.fit(X)
X = sc.transform(X)
'''

# Read Test Data
Test = pd.read_csv('./data/test' + str(rank + 1) + '.csv')

# Construct and predict
params = [3, 6, None]

# single tree in each process
'''
rt, error, score = single_regression_tree(X, y, params)
print("In Tree " + str(rank + 1) + ", validation MSE error = " + str(error) + ", r2_score = " + str(score))
'''    

# mutilple tree in each process
rtArr, test_errorArr, test_scoreArr, train_errorArr, train_scoreArr = single_regression_tree(X, y, params)
for i in range(0, 150):
    print("In Process " + str(rank + 1) + ", No." + str(i) + "tree" + ", train MSE error = " + str(train_errorArr[i]) + ", r2score = " + str(train_scoreArr[i]) + ", validation MSE error = " + str(test_errorArr[i]) + ", r2_score = " + str(test_scoreArr[i]))
for i in range(len(rtArr)):
    joblib.dump(rtArr[i], './model/RT' + str(rank) + '_' +  str(i) + '.model')
    
# send to the root
# joblib.dump(rt, './model/regressiontree' + str(rank + 1) + '.model')

# Visulize one model
dot_data = tree.export_graphviz(rtArr[0], out_file=None)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf('./result/test_model.pdf')