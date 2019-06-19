import pandas as pd
import os
data1 = pd.read_csv('./result/result_1.csv')
data1.to_csv('./result/final_9.csv', header=1, index=False, mode='a')

data1 = pd.read_csv('./result/result_2.csv', header=1)
data1.to_csv('./result/final_9.csv', header=1, index=False, mode='a')

data1 = pd.read_csv('./result/result_3.csv', header=1)
data1.to_csv('./result/final_9.csv', header=1, index=False, mode='a')

data1 = pd.read_csv('./result/result_4.csv', header=1)
data1.to_csv('./result/final_9.csv', header=1, index=False, mode='a')

data1 = pd.read_csv('./result/result_5.csv', header=1)
data1.to_csv('./result/final_9.csv', header=1, index=False, mode='a')

data1 = pd.read_csv('./result/result_6.csv', header=1)
data1.to_csv('./result/final_9.csv', header=1, index=False, mode='a')