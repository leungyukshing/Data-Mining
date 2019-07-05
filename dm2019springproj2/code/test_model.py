import numpy as np
from mpi4py import MPI
import pandas as pd

from sklearn import tree

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from sklearn.externals import joblib
# model visulazation
from sklearn.externals.six import StringIO
import pydotplus
from sklearn.tree import export_graphviz
from IPython.display import Image


'''
  This script is used to check predition and visualize model. You can just ignore it.
'''

model1 = joblib.load('./model/regressiontree1.model')
prediction1 = model1.predict([[-0.50568, 2, -218, -15, 151, -1, -74, -0.49038, 0.4587, 0.46732, -0.47841, 0.99324, 0.48148]])
print(prediction1)

model2 = joblib.load('./model/regressiontree2.model')
prediction2 = model2.predict([[-0.50568, 2, -218, -15, 151, -1, -74, -0.49038, 0.4587, 0.46732, -0.47841, 0.99324, 0.48148]])
print(prediction2)

model3 = joblib.load('./model/regressiontree3.model')
prediction3 = model3.predict([[-0.50568, 2, -218, -15, 151, -1, -74, -0.49038, 0.4587, 0.46732, -0.47841, 0.99324, 0.48148]])
print(prediction3)

model4 = joblib.load('./model/regressiontree4.model')
prediction4 = model4.predict([[-0.50568, 2, -218, -15, 151, -1, -74, -0.49038, 0.4587, 0.46732, -0.47841, 0.99324, 0.48148]])
print(prediction4)

model5 = joblib.load('./model/regressiontree5.model')
prediction5 = model5.predict([[-0.50568, 2, -218, -15, 151, -1, -74, -0.49038, 0.4587, 0.46732, -0.47841, 0.99324, 0.48148]])
print(prediction5)


def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=0)

loss = [0.5, 0.21, 0.4, 0.37, 0.27]
loss = np.ones(5) - loss
after_softmax = softmax(loss)
print(after_softmax)

final1 = prediction1 + prediction2 + prediction3 + prediction4 + prediction5
final1 /= 5
print(final1)


final2 = prediction1[0] * after_softmax[0] + prediction2[0] * after_softmax[1] + prediction3[0] * after_softmax[2] + prediction4[0] * after_softmax[3] + prediction5 * after_softmax[4]
print(final2)

features = np.arange(0, 13, 1)

dot_data = StringIO()
export_graphviz(model1, feature_names=features, out_file=dot_data, filled=True, rounded=True, special_characters=True)
(graph,) = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())
print("save file")
graph.write_pdf('./model/model1.pdf')
