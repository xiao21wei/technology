import os

import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import LSTM, Dense
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def lstm_model_plus_test():
    # 读取数据
    data = pd.read_csv('Ng_GenPCal.csv')

    data = data.iloc[::10, :]

    # 提取需要的特征和目标变量
    features = ['time', 'Ng', 'GenPCal']
    target_variable = 'GenPCal'
    data = data[features]

    # 将时间列转换为日期时间格式
    data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S.%f')

    # 数据归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.drop(columns='time'))

    # 定义时间步长（用于定义输入序列长度和输出序列长度）
    time_steps = 30

    # 定义训练集和测试集的比例
    test_split = round(len(data) * 0.20)

    # 定义训练集和测试集
    train_data = scaled_data[:-test_split]
    test_data = scaled_data[-test_split:]

    print(test_data.shape)

    # 准备训练数据
    X_train, y_train = [], []
    for i in range(time_steps, len(train_data)):
        X_train.append(train_data[i - time_steps:i, :])
        y_train.append(train_data[i, 1])

    X_train, y_train = np.array(X_train), np.array(y_train)

    # 准备测试数据
    X_test, y_test = [], []
    for i in range(time_steps, len(test_data)):
        X_test.append(test_data[i - time_steps:i, :])
        y_test.append(test_data[i, 1])

    X_test, y_test = np.array(X_test), np.array(y_test)

    # 构建LSTM模型
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(units=50))
    model.add(Dense(1))

    # 编译模型
    model.compile(loss='mean_squared_error', optimizer='adam')

    # 训练模型
    model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test), verbose=1, shuffle=False)

    # 保存模型
    # model.save('lstm_model_plus.h5')

    # 预测
    y_pred = model.predict(X_test)
    print(y_pred.shape)
    print(y_test.shape)
    # 反归一化
    y_pred_copies_array = np.repeat(y_pred, 2, axis=-1)
    y_pred = scaler.inverse_transform(np.reshape(y_pred_copies_array, (len(y_pred), 2)))[:, 1]
    y_test_copies_array = np.repeat(y_test, 2, axis=-1)
    y_test = scaler.inverse_transform(np.reshape(y_test_copies_array, (len(y_test), 2)))[:, 1]

    # 保存测试集的真实值和预测值存入data.csv文件
    data = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred.reshape(-1)})
    data.to_csv('data.csv', index=False)

    # 将预测数据和测试数据绘制在一张图中
    plt.plot(y_test, label='Real GenPCal')
    plt.plot(y_pred, label='Predicted GenPCal')
    plt.title('GenPCal Prediction')
    plt.xlabel('Time')
    plt.ylabel('GenPCal')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    lstm_model_plus_test()
