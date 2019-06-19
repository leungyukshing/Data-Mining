import pandas as pd
import os

data = pd.read_csv('./data/train1.csv', header=None)
data.to_csv('./data/all_train.csv', mode='a', index=False, header=None)

data = pd.read_csv('./data/train2.csv', header=None)
data.to_csv('./data/all_train.csv', mode='a', index=False, header=None)

data = pd.read_csv('./data/train3.csv', header=None)
data.to_csv('./data/all_train.csv', mode='a', index=False, header=None)

data = pd.read_csv('./data/train4.csv', header=None)
data.to_csv('./data/all_train.csv', mode='a', index=False, header=None)

data = pd.read_csv('./data/train5.csv', header=None)
data.to_csv('./data/all_train.csv', mode='a', index=False, header=None)


label = pd.read_csv('./data/label1.csv', header=None)
label.to_csv('./data/all_label.csv', mode='a', index=False, header=None)
label = pd.read_csv('./data/label2.csv', header=None)
label.to_csv('./data/all_label.csv', mode='a', index=False, header=None)
label = pd.read_csv('./data/label3.csv', header=None)
label.to_csv('./data/all_label.csv', mode='a', index=False, header=None)
label = pd.read_csv('./data/label4.csv', header=None)
label.to_csv('./data/all_label.csv', mode='a', index=False, header=None)
label = pd.read_csv('./data/label5.csv', header=None)
label.to_csv('./data/all_label.csv', mode='a', index=False, header=None)