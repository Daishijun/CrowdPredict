# -*- coding: utf-8 -*-
# @Date    : 2019/3/20
# @Time    : 13:38
# @Author  : Daishijun
# @File    : airpassenger.py
# Software : PyCharm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR


from keras.utils.vis_utils import model_to_dot

np.random.seed(7)

def preprocess(trainX=None, testX=None, preprocessing = 'mm', redataf = True):
    if preprocessing == 'mm':
        prep = MinMaxScaler()
    elif preprocessing == 'st':
        prep = StandardScaler()
    else:
        raise Exception('invalid preprocessing', preprocessing)
    if trainX is not None:
        trainX = prep.fit_transform(trainX)
        if redataf:
            trainX = pd.DataFrame(trainX, columns=trainX.columns.values.tolist())
    if testX is not None:
        testX = prep.transform(testX)
        if redataf:
            testX = pd.DataFrame(testX, columns=testX.columns.values.tolist())
    return trainX, testX, prep


def create_dataset(dataset, look_back=1):
    '''

    :param dataset:
    :param look_back:
    :return:
    '''
    dataX = []; dataY =[]
    for i in range(len(dataset)-look_back):
        a = dataset[i:(i+look_back),0]
        dataX.append(a)
        dataY.append(dataset[i+look_back,0])
    return np.array(dataX, dtype=np.float32), np.array(dataY, dtype=np.float32)



def LSTMtest(dataset, rnn_units=4, epochs=50):
    np.random.seed(7)
    train_size = 70

    test_size = len(dataset) - train_size
    _train, _test = dataset[0:train_size, :], dataset[train_size:, :]
    print(_train.shape, _test.shape)
    train, test, scaler = preprocess(_train, _test, redataf=False)

    look_back = 7
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    print(trainX)

    # trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    # testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    print(trainX.shape, testX.shape)

    model = Sequential()
    model.add(LSTM(rnn_units, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=epochs, batch_size=1, verbose=1)

    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])

    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))

    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))

    #plot
    fig = plt.figure()

    ax1 = fig.add_subplot(321)  # 3行2列第一块
    ax1.set_title('TrainSet')
    ax1.plot(trainY[0], c='r', label='TrainY')
    ax1.plot(trainPredict[:, 0], c='g', label='TrainPredict')
    ax1.legend()

    ax2 = fig.add_subplot(322)  # 2行1列第二块
    ax2.set_title('TestSet')
    ax2.plot(testY[0], c='r', label='TestY')
    ax2.plot(testPredict[:, 0], c='g', label='TestPredict')
    ax2.legend()

    #add SVR resmodel
    # print(trainX.shape)
    trainResX = np.reshape(trainX, (trainX.shape[0], trainX.shape[2]))
    trainResY_raw = trainY[0]-trainPredict[:,0]

    testResX = np.reshape(testX, (testX.shape[0], testX.shape[2]))
    #计算残差
    testResY_raw = testY[0]-testPredict[:,0]  #真实-LSTM预测
    trainResY_raw =np.reshape(trainResY_raw.astype('float32'), (len(trainResY_raw), -1))

    #未能归一化？？
    # trainResY_scale, _,svrscaler = preprocess(trainX=trainResY_raw,redataf=False)
    trainResY_scale, _,svrscaler = preprocess(trainX=trainResY_raw,redataf=False)


    rbf_svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5, \
                           param_grid={'C': [1e-1, 1e0, 1e1, 1e2], 'gamma': np.logspace(-2, 2, 10)})


    rbf_svr.fit(trainResX, trainResY_scale[:,0])    #输入与LSTM的输入相同，为第一次归一化的数据； label为LSTM的输出反归一化后的残差，在经过归一化的到的值

    resTrain_svrPre = rbf_svr.predict(trainResX)    #此处用的就是第一步归一化的，等同于LSTM的输入
    resTest_svrPre = rbf_svr.predict(testResX)


    trainResY_scale = svrscaler.inverse_transform([trainResY_scale[:,0]])  #SVR训练集groudtruth的反归一化

    resTest_svrPre = svrscaler.inverse_transform([resTest_svrPre])  # SVR测试集预测的反归一化
    resTrain_svrPre = svrscaler.inverse_transform([resTrain_svrPre])     #SVR训练集预测的反归一化



    #plot SVR 对于残差的拟合
    ax3 = fig.add_subplot(323)
    ax3.plot(trainResY_scale[0], c='r', label='Res True')
    ax3.plot(resTrain_svrPre[0], c ='g', label='Res Pre')
    ax3.set_title('SVR train')
    ax3.legend()

    ax4 = fig.add_subplot(324)
    ax4.plot(testResY_raw, c='r', label='Res True')
    ax4.plot(resTest_svrPre[0], c='g', label='Res Pre')
    ax4.set_title('SVR test')
    ax4.legend()


    fixedTrainY = trainPredict[:,0]+resTrain_svrPre[0]    #修正后的结果  LSTM预测 + SVR预测（真实-LSTM预测）
    fixedTestY = testPredict[:,0]+resTest_svrPre[0]

    #plot
    ax5 = fig.add_subplot(325)
    ax5.plot(trainY[0], c='r', label='fixed True')
    ax5.plot(fixedTrainY, c='g', label='fixed Pre')
    ax5.set_title('L+S train')
    ax5.legend()

    ax6 = fig.add_subplot(326)
    ax6.plot(testY[0], c='r', label='True')
    ax6.plot(fixedTestY, c='g', label='Pre')
    ax6.set_title('L+S test')
    ax6.legend()

    #fixed Score
    fixed_trainScore = math.sqrt(mean_squared_error(trainY[0], fixedTrainY))

    fixed_testScore = math.sqrt(mean_squared_error(testY[0], fixedTestY))

    plt.show()
    plt.close()




    return trainScore, testScore, fixed_trainScore, fixed_testScore

if __name__ == '__main__':
    resultdict = []
    df = pd.read_csv(r'D:\Users\ICT_DSJ\DMdataAnalyze\flow\L1\choosedtimadata.csv', usecols=[1], engine='python')
    # df = pd.read_csv(r'D:\Users\DSJ\Documents\shijianxulie\AirPassengers.csv', usecols=[1], engine='python')

    dataset = df.values[21:-7].astype('float32')
    # dataset = df.values[:].astype('float32')

    rnn_units=4
    for epoch in range(5,10,1):
        trainS, testS, fixed_trainS, fixed_testS = LSTMtest(dataset, epochs=epoch)
        resultdict.append({'rnn_unit':rnn_units, 'epochs':epoch, 'trainScore':trainS, 'testScore':testS, 'fixed_trainScore':fixed_trainS, 'fixed_testScore':fixed_testS})
    for li in resultdict:
        print(li)
