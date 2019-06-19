import numpy as np
from mpi4py import MPI
import pandas as pd
# Get MPI info
'''
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    avg = np.zeros(5)
    total = []
    for i in range(0, 4):
        data_recv = comm.recv(source = i+1)
        print("root receive from " + str(i+1) + ", ")
        print(data_recv)
        avg += data_recv
        total.append(data_recv)
    
    print(avg)
else:
    data_send = [rank] * 5
    comm.send(data_send, dest = 0)
    print("my rank is " + str(rank) + ", send data" )
    print(data_send)
'''
'''
def divive(c):
    return c / 5

l1 = [1, 2, 3, 4, 5]
l3 = [2, 3, 4, 5, 6]
l3 += l1
print(l3)
l2 = [c / 5 for c in l1]
print(l2)
'''

'''
# check train data size
train1 = pd.read_csv('./data/train1.csv', header=None)
print(train1.shape)
train2 = pd.read_csv('./data/train2.csv', header=None)
print(train2.shape)
train3 = pd.read_csv('./data/train3.csv', header=None)
print(train3.shape)
train4 = pd.read_csv('./data/train4.csv', header=None)
print(train4.shape)
train5 = pd.read_csv('./data/train5.csv', header=None)
print(train5.shape)

print('--------------------------')

# check test data size
data1 = pd.read_csv('./data/test1.csv', header=None)
print(data1.shape)
data2 = pd.read_csv('./data/test2.csv', header=None)
print(data2.shape)
data3 = pd.read_csv('./data/test3.csv', header=None)
print(data3.shape)
data4 = pd.read_csv('./data/test4.csv', header=None)
print(data4.shape)
data5 = pd.read_csv('./data/test5.csv', header=None)
print(data5.shape)
data6 = pd.read_csv('./data/test6.csv', header=None)
print(data6.shape)


'''

data1 = pd.read_csv('./test/t1.csv')
data1.to_csv('./test/result.csv', header=1, index=False, mode='a')

data1 = pd.read_csv('./test/t2.csv', header=1)

data1.to_csv('./test/result.csv', header=1, index=False, mode='a')
