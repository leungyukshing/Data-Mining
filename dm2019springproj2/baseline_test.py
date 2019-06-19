import numpy as np
from mpi4py import MPI
import pandas as pd

from sklearn import tree

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from sklearn.externals import joblib

import warnings
warnings.filterwarnings("ignore")

# Get MPI info
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if (size != 36):
    print("Error!")
    exit()

def test(X, path):
    print("In test function")
    regressionTree = joblib.load(path)
    prediction = regressionTree.predict(X)
    print("Test finished")
    return prediction

def list_add(a,b):
    c = []
    for i in range(len(a)):
        c.append(a[i]+b[i])
    return c

test_file_length = [2000001, 2000001, 2000001, 2000001, 2000001, 915116]
start_ID = [1, 2000002, 4000003, 6000004, 8000005, 10000006]
# 6 processes, each process one test file

if rank >= 0 and rank <= 5:
    # Per Test
    avg = np.zeros(test_file_length[rank])
    for j in range(0, 5):
        data_recv = comm.recv(source = (rank + 1) * 5 + 1 + j)
        print("In rank " + str(rank) + " receive " + str((rank + 1) * 5 + 1 + j))
        # ERROR!!
        # avg += data_recv
        print(type(data_recv))
        print(type(avg))
        print(len(data_recv))
        print(len(avg))
        avg = list_add(avg, data_recv)
        
    print("In rank = " + str(rank) + " final receive ")
    

    # avg /= 5
    avg = [c / 5 for c in avg]
    #print(avg)
    n = len(avg)
    avg_array = np.array(avg)
    result = pd.DataFrame({'Id': list(range(start_ID[rank], start_ID[rank] + test_file_length[rank])), 'Predicted': avg_array.astype(np.float64)})
    result.to_csv('./result/result_' + str(rank + 1) + '.csv', float_format='%lf', index=False)
else:
    # each process test seperatly
    if (rank >= 6 and rank <= 10):
        X = pd.read_csv('./data/test1.csv', header=None)
    elif (rank >= 11 and rank <= 15):
        X = pd.read_csv('./data/test2.csv', header=None)
    elif (rank >= 16 and rank <= 20):
        X = pd.read_csv('./data/test3.csv', header=None)
    elif (rank >= 21 and rank <= 25):
        X = pd.read_csv('./data/test4.csv', header=None)
    elif (rank >= 26 and rank <= 30):
        X = pd.read_csv('./data/test5.csv', header=None)
    elif (rank >= 31 and rank <= 35):
        X = pd.read_csv('./data/test6.csv', header=None)
    
    modelNum = rank % 5
    if modelNum == 0:
        modelNum = 5
    path = './model/regressiontree' + str(modelNum) + '.model'
    
    # predict
    res = np.zeros(len(X))
    for i in range(150):
        path = './model/RT' + str(modelNum - 1) + '_' + str(i) + '.model'
        pred = test(X, path)
        res = list_add(res, pred)
    # pred = test(X, path)
    res = [c / 150 for c in res]

    # send to master
    if (rank >= 6 and rank <= 10):
        comm.send(res, dest=0)
        print("send from " + str(rank) + " to 0")
    elif (rank >= 11 and rank <= 15):
        comm.send(res, dest=1)
        print("send from " + str(rank) + " to 1")
    elif (rank >= 16 and rank <= 20):
        comm.send(res, dest=2)
        print("send from " + str(rank) + " to 2")
    elif (rank >= 21 and rank <= 25):
        comm.send(res, dest=3)
        print("send from " + str(rank) + " to 3")
    elif (rank >= 26 and rank <= 30):
        comm.send(res, dest=4)
        print("send from " + str(rank) + " to 4")
    elif (rank >= 31 and rank <= 35):
        comm.send(res, dest=5)
        print("send from " + str(rank) + " to 5")