# 构建lstm模型，进行多变量时序预测,Ng_GenPCal.csv文件中的数据为：time,Ng,GenPCal,需要预测的值为GenPCal
import os

import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import LSTM, Dropout, Dense
from keras.wrappers.scikit_learn import KerasRegressor
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import keras.backend as k


def positive_mse(y_true, y_pred):  # 定义损失函数，使得预测值为正，即不考虑负值，只考虑预测值大于0的情况，这样可以提高预测的准确性
    return k.mean(k.square(k.maximum(y_pred, 0) - k.maximum(y_true, 0)), axis=-1)


def lstm_model_pro_test():
    df = pd.read_csv('Ng_GenPCal.csv', parse_dates=['time'], index_col=[0])
    print(df.shape)

    # df = df.iloc[::100, :]

    # 处理缺失值
    df = df.dropna()
    # 调整数据集索引
    df = df.sort_index()

    test_split = round(len(df) * 0.20)
    df_for_training = df[:-test_split]
    df_for_testing = df[-test_split:]
    print(df_for_training.shape)
    print(df_for_testing.shape)

    scaler = MinMaxScaler(feature_range=(0, 1))
    df_for_training_scaled = scaler.fit_transform(df_for_training)
    df_for_testing_scaled = scaler.transform(df_for_testing)

    trainX, trainY = createXY(df_for_training_scaled, 50)
    testX, testY = createXY(df_for_testing_scaled, 50)

    # trainX, trainY = createXY(df_for_training.values, 30)
    # testX, testY = createXY(df_for_testing.values, 30)

    print("trainX Shape-- ", trainX.shape)
    print("trainY Shape-- ", trainY.shape)

    print("testX Shape-- ", testX.shape)
    print("testY Shape-- ", testY.shape)

    # 遍历trainX,trainY,testX,testY,输出其中的负值
    for i in range(len(trainX)):
        for j in range(len(trainX[i])):
            if trainX[i][j][0] < 0:
                print("trainX[", i, "][", j, "][0]=", trainX[i][j][0])
            if trainX[i][j][1] < 0:
                print("trainX[", i, "][", j, "][1]=", trainX[i][j][1])
        if trainY[i] < 0:
            print("trainY[", i, "]=", trainY[i])

    for i in range(len(testX)):
        for j in range(len(testX[i])):
            if testX[i][j][0] < 0:
                print("testX[", i, "][", j, "][0]=", testX[i][j][0])
            if testX[i][j][1] < 0:
                print("testX[", i, "][", j, "][1]=", testX[i][j][1])
        if testY[i] < 0:
            print("testY[", i, "]=", testY[i])

    # grid_model = KerasRegressor(build_fn=build_model, verbose=1, validation_data=(testX, testY))
    #
    # parameters = {
    #     'batch_size': [128],
    #     'epochs': [40, 50],
    #     'optimizer': ['SGD']
    # }
    #
    # grid_search = GridSearchCV(
    #     estimator=grid_model,
    #     param_grid=parameters,
    #     cv=5,
    #     scoring='neg_mean_squared_error',
    #     verbose=1)
    # grid_search = grid_search.fit(trainX, trainY)
    #
    # # 输出最优的参数组合
    # print(grid_search.best_params_)
    #
    # my_model = grid_search.best_estimator_.model

    parameters = {'batch_size': 128,  # 批处理大小
                  'epochs': 50,  # 迭代次数
                  'optimizer': 'SGD'  # 优化器
                  }
    # # 使用parameters中的参数构建模型
    my_model = build_model(parameters['optimizer'])
    # 设置模型的batch_size和epochs
    my_model.fit(trainX, trainY, batch_size=parameters['batch_size'], epochs=parameters['epochs'])

    my_model.save('my_model.h5')

    my_model = load_model('my_model.h5')

    prediction = my_model.predict(testX)
    # prediction1 = my_model.predict(trainX)
    # pred = prediction[:, 0]
    # original = testY

    prediction_copies_array = np.repeat(prediction, 2, axis=-1)
    print(prediction_copies_array.shape)

    pred = scaler.inverse_transform(np.reshape(prediction_copies_array, (len(prediction), 2)))[:, 1]

    original_copies_array = np.repeat(testY, 2, axis=-1)
    original = scaler.inverse_transform(np.reshape(original_copies_array, (len(testY), 2)))[:, 1]

    # # 将训练数据和测试数据绘制在一张图中
    # plt.plot(df_for_training['GenPCal'], label='Training Data')
    # plt.plot(df_for_testing['GenPCal'], label='Testing Data')
    # plt.legend()
    # plt.show()

    # 将预测结果和测试结果输出到result.csv文件中
    result = pd.DataFrame({'prediction': pred, 'original': original})
    result.to_csv('result.csv', index=False)

    # 输出参数和预测结果和测试结果的MSE
    print(parameters)
    print("MSE-- ", np.mean(np.square(pred - original)))

    # 将预测数据和测试数据绘制在一张图中
    plt.plot(original, label='Original Data')
    plt.plot(pred, label='Predicted Data')
    plt.legend()
    plt.show()


def createXY(dataset, n_past):
    dataX = []
    dataY = []
    for i in range(n_past, len(dataset)):
        # dataX为前n_past天的Ng数据
        # dataY为第n_past天的GenPCal数据
        dataX.append(dataset[i - n_past:i, 0:dataset.shape[1]])
        dataY.append(dataset[i, 1])
    return np.array(dataX), np.array(dataY)


def build_model(optimizer):  # 构建模型，optimizer为优化器
    grid_model = Sequential()
    grid_model.add(LSTM(50, return_sequences=True, input_shape=(50, 2)))
    grid_model.add(LSTM(50))  # return_sequences默认为False
    grid_model.add(Dropout(0.2))  # 防止过拟合
    grid_model.add(Dense(1, activation='relu'))  # 全连接层

    grid_model.compile(loss='mse', optimizer=optimizer)  # 编译模型
    # grid_model.compile(loss=positive_mse, optimizer=optimizer)  # 编译模型，使用自定义的损失函数
    return grid_model


if __name__ == '__main__':
    lstm_model_pro_test()
