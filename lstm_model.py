import numpy as np
import pandas as pd
from datetime import datetime

from keras import Sequential
from keras.layers import Dropout, LSTM, Dense, GRU
from keras.optimizers import SGD
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def lstm_model_test(csv_file, steps):
    # Load the data
    custom_date_parser = lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
    df = pd.read_csv(csv_file, parse_dates=['trendTime'], date_parser=custom_date_parser, index_col='trendTime')
    # Rename the columns
    last_data_time = df.index[-1]
    # last_data_time为%Y-%m-%d %H:%M:%S格式,计算半天前的时间
    last_data_time = last_data_time - pd.Timedelta(hours=12)
    # 划分训练集和测试集
    train_df = df[df.index < last_data_time].values
    test_df = df[df.index >= last_data_time].values

    sc = MinMaxScaler(feature_range=(0, 1))
    train_df_scaled = sc.fit_transform(train_df)

    x_train = []
    y_train = []
    for i in range(steps, train_df_scaled.shape[0]):
        x_train.append(train_df_scaled[i-steps:i, 0])
        y_train.append(train_df_scaled[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # model = Sequential()
    # model.add(LSTM(units=128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    # model.add(Dropout(0.2))
    #
    # model.add(LSTM(units=128, return_sequences=True))
    # model.add(Dropout(0.2))
    #
    # model.add(LSTM(units=128))
    # model.add(Dropout(0.2))
    #
    # model.add(Dense(units=1))
    #
    # model.compile(optimizer='rmsprop', loss='mse')
    #
    # model.fit(x_train, y_train, epochs=20, batch_size=32)
    #
    # df = pd.concat((df[len(df) - len(test_df) - steps:], df[len(df) - len(test_df):]), axis=0)
    # inputs = df[len(df) - len(test_df) - steps:].values
    # inputs = inputs.reshape(-1, 1)
    # inputs = sc.transform(inputs)
    # x_test = []
    # for i in range(steps, inputs.shape[0]):
    #     x_test.append(inputs[i-steps:i, 0])
    # x_test = np.array(x_test)
    #
    # x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    #
    # predict_test = model.predict(x_test)  # 预测
    # predict_test = sc.inverse_transform(predict_test)  # 反归一化

    # # 将训练数据，测试数据，预测数据呈现在一张图上
    # plt.plot(test_df, label='test')
    # plt.plot(predict_test, label='predict')
    # # 添加标题
    # plt.title("lstm(steps:" + str(steps) + ")")
    # plt.legend()
    # plt.show()
    #
    # # 将生成的图像保存至本地
    # plt.savefig("lstm_" + str(steps) + ".png")

    model_gru = Sequential()
    model_gru.add(GRU(50, return_sequences=True, input_shape=(x_train.shape[1], 1), activation='tanh'))
    model_gru.add(Dropout(0.2))
    model_gru.add(GRU(50, activation='tanh'))
    model_gru.add(Dropout(0.2))
    model_gru.add(Dense(1))

    model_gru.compile(optimizer=SGD(lr=0.01, decay=1e-7, momentum=0.9), loss='mse')
    # 模型的训练
    model_gru.fit(x_train, y_train, epochs=20, batch_size=32)

    df = pd.concat((df[len(df) - len(test_df) - steps:], df[len(df) - len(test_df):]), axis=0)
    inputs = df[len(df) - len(test_df) - steps:].values
    inputs = inputs.reshape(-1, 1)
    inputs = sc.transform(inputs)
    x_test = []
    for i in range(steps, inputs.shape[0]):
        x_test.append(inputs[i-steps:i, 0])
    x_test = np.array(x_test)

    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predict_test = model_gru.predict(x_test)
    predict_test = sc.inverse_transform(predict_test)

    # 将训练数据，测试数据，预测数据呈现在一张图上
    plt.plot(test_df, label='test')
    plt.plot(predict_test, label='predict')
    # 添加标题
    plt.title("GRU(steps:"+str(steps)+")")
    plt.legend()
    plt.show()

    # 将生成的图像保存至本地
    plt.savefig("GRU_"+str(steps)+".png")


if __name__ == "__main__":
    for i in range(3, 30):
        lstm_model_test("1Aa.csv", i)
