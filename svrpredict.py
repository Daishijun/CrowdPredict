# -*- coding: utf-8 -*-
# @Date    : 2019/3/21
# @Time    : 16:42
# @Author  : Daishijun
# @File    : svrpredict.py
# Software : PyCharm

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from airpassenger import create_dataset, preprocess
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
import math
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # df = pd.read_csv(r'D:\Users\ICT_DSJ\DMdataAnalyze\flow\L1\choosedtimadata.csv', usecols=[1], engine='python')
    df = pd.read_csv(r'D:\Users\DSJ\Documents\shijianxulie\AirPassengers.csv', usecols=[1], engine='python')
    # dataset = df.values[21:-7].astype('float32')
    dataset = df.values[:].astype('float32')

    train_size = 100

    test_size = len(dataset) - train_size
    _train, _test = dataset[0:train_size, :], dataset[train_size:, :]
    print(_train.shape, _test.shape)
    train, test, scaler = preprocess(_train, _test, redataf=False)

    look_back = 7
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    # rbf_svr = SVR(kernel='rbf')
    rbf_svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5, param_grid={'C':[1e-1,1e0, 1e1, 1e2], 'gamma': np.logspace(-2,2,10)})

    rbf_svr.fit(trainX, trainY)
    print('best params:', rbf_svr.best_params_)


    testYpred = rbf_svr.predict(testX)

    testY = scaler.inverse_transform([testY])
    testYpred = scaler.inverse_transform([testYpred])
    # print(type(testYpred))
    # print(testYpred)
    # print(type(testY))
    #
    # print(testY[0])
    # print(testYpred[0])
    testScore = math.sqrt(mean_squared_error(testY[0], testYpred[0]))
    print('score:', testScore)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(testY[0], c='r', label='True')
    ax1.plot(testYpred[0], c = 'g', label='Predict')
    plt.legend()
    plt.show()

    print('差值:',testY[0]-testYpred[0])




