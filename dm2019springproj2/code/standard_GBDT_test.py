import pandas as pd
import numpy as np
from sklearn.externals import joblib

test_file_length = [2000001, 2000001, 2000001, 2000001, 2000001, 915116]
start_ID = [1, 2000002, 4000003, 6000004, 8000005, 10000006]

model = joblib.load('../model/sklearn_GBDT_model.model')
for i in range(1, 7):
    X = pd.read_csv('../data/test' + str(i) + '.csv', header=None)
    pred = model.predict(X)


    result = pd.DataFrame({'Id': list(range(start_ID[i-1], start_ID[i-1] + test_file_length[i-1])), 'Predicted': pred.astype(np.float64)})
    result.to_csv('../result/standard/result_' + str(i) + '.csv', float_format='%lf', index=False)
    print('finish ' + str(i))

