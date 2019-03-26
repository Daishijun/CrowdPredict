# -*- coding: utf-8 -*-
# @Date    : 2019/3/22
# @Time    : 18:39
# @Author  : Daishijun
# @File    : SvrAndLstm.py
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
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM



def svrModel(dataset, fig):
    train_size = 70
    test_size = len(dataset) - train_size
    _train, _test = dataset[0:train_size, :], dataset[train_size:, :]
    train, test, scaler = preprocess(_train, _test, redataf=False)     #用训练集归一化后的训练数据和测试数据
    look_back = 3
    trainX, trainY = create_dataset(train, look_back)     #shape:(63*7), (63,)
    testX, testY = create_dataset(test, look_back)
    rbf_svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5,
                           param_grid={'C': [1e-1, 1e0, 1e1, 1e2], 'gamma': np.logspace(-2, 2, 10)})
    rbf_svr.fit(trainX, trainY)
    print('best params:', rbf_svr.best_params_)
    testYpred = rbf_svr.predict(testX)

    trainY = scaler.inverse_transform([trainY])
    trainYpred = rbf_svr.predict(trainX)    #训练集的预测Y，为了求出与真实值的残差，用于训练LSTM
    trainYpred = scaler.inverse_transform([trainYpred])
    trainY_Res = trainY - trainYpred    #真实-预测    shape（1*63）

    testY = scaler.inverse_transform([testY])    #反归一化的真实testY
    testYpred = scaler.inverse_transform([testYpred])    #反归一化的真实test预测值
    testY_Res = testY - testYpred    #测试集 真实-预测    shape （1*35）

    trainScore = math.sqrt(mean_squared_error(trainY[0], trainYpred[0]))
    testScore = math.sqrt(mean_squared_error(testY[0], testYpred[0]))
    # print('score:', testScore)

    ax1 = fig.add_subplot(321)
    ax1.set_title('SVR TrainSet')
    ax1.plot(trainY[0], c='r', label='TrainY')
    ax1.plot(trainYpred[0], c='g', label='TrainPredict')
    ax1.legend()

    ax2 = fig.add_subplot(322)
    ax2.set_title('SVR TestSet')
    ax2.plot(testY[0], c='r', label='TestY')
    ax2.plot(testYpred[0], c='g', label='TestPredict')
    ax2.legend()
    # plt.show()
    # plt.close()


    return  [trainX, trainY_Res, testX, testY_Res], [trainY[0],trainYpred[0],testY[0],testYpred[0]] , [trainScore,testScore]   #（63*7）normal；（1*63）；（35*7）normal；（1*37） ; 后面一个是【训练集真实Y，SVR预测Y，测试真实Y,SVR预测】

def lstmRes(dataset,resultSVR,rnn_units=4, epochs=5):
    np.random.seed(7)
    look_back = 3
    [trainX, trainY, testX, testY] = dataset
    [trainY_truth, svrPredtrain, testY_truth, svrPredtest] = resultSVR
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    trainY_scaled, _, scaler = preprocess(trainY.T, redataf=False)    #将Y值归一化


    model = Sequential()
    model.add(LSTM(rnn_units, input_shape=(1, look_back)))

    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY_scaled, epochs=epochs, batch_size=1, verbose=1)

    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    trainPredict = scaler.inverse_transform(trainPredict)    #训练集预测值

    testPredict = scaler.inverse_transform(testPredict)      #测试集预测值

    trainScore_Res = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))

    testScore_Res = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
    #plot
    ax3 = fig.add_subplot(323)  # 3行2列第3块
    ax3.set_title('LSTM TrainSet')
    ax3.plot(trainY[0], c='r', label='TrainY')
    ax3.plot(trainPredict[:, 0], c='g', label='TrainPredict')
    ax3.legend()

    ax4 = fig.add_subplot(324)  # 2行1列第4块
    ax4.set_title('LSTM TestSet')
    ax4.plot(testY[0], c='r', label='TestY')
    ax4.plot(testPredict[:, 0], c='g', label='TestPredict')
    ax4.legend()
    # plt.show()
    # plt.close()

    trainY_all = trainPredict[:,0]+ svrPredtrain
    testY_all = testPredict[:,0]+ svrPredtest
    fixed_trainScore = math.sqrt(mean_squared_error(trainY_truth, trainY_all))
    fixed_testScore = math.sqrt(mean_squared_error(testY_truth, testY_all))
    #plot

    ax5 = fig.add_subplot(325)
    ax5.plot(trainY_truth, c='r', label='fixed True')
    ax5.plot(trainY_all, c='g', label='fixed Pre')
    ax5.set_title('L+S train')
    ax5.legend()


    ax6 = fig.add_subplot(326)
    ax6.plot(testY_truth, c='r', label='True')
    ax6.plot(testY_all, c='g', label='Pre')
    ax6.set_title('L+S test')
    ax6.legend()
    # plt.show()
    # plt.close()

    return trainScore_Res, testScore_Res, fixed_trainScore,fixed_testScore










if __name__ == '__main__':
    df = pd.read_csv(r'D:\Users\ICT_DSJ\DMdataAnalyze\flow\L1\choosedtimadata.csv', usecols=[1], engine='python')
    # df = pd.read_csv(r'D:\Users\DSJ\Documents\shijianxulie\AirPassengers.csv', usecols=[1], engine='python')
    dataset = df.values[21:-7].astype('float32')


    resultdict = []
    rnn_units = 8

    # lstmRes(dataforLSTM, resultSVR)
    for epoch in range(100,110,2):
        fig = plt.figure()
        dataforLSTM, resultSVR, [trainS,testS] = svrModel(dataset, fig)
        _res, _testres, fixed_trainS, fixed_testS = lstmRes(dataforLSTM, resultSVR,rnn_units=rnn_units,epochs=epoch)
        resultdict.append({'rnn_unit':rnn_units, 'epochs':epoch, 'trainScore':trainS, 'testScore':testS, 'fixed_trainScore':fixed_trainS, 'fixed_testScore':fixed_testS})
    for li in resultdict:
        print(li)


