import numpy as np
import pandas as pd
from datetime import datetime

from keras import Sequential
from keras.layers import Dropout, LSTM, Dense
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def lstm_model_test(csv_file, steps):
    # Load the data
    custom_date_parser = lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
    df = pd.read_csv(csv_file, parse_dates=['trendTime'], date_parser=custom_date_parser, index_col='trendTime')
    # Rename the columns
    last_data_time = df.index[-1]

    start_time = pd.to_datetime(last_data_time) - pd.Timedelta(days=1) - pd.Timedelta(hours=1)
    start_time = start_time.strftime('%Y-%m-%d %H:%M:%S')
    print(start_time)

    # last_data_time为%Y-%m-%d %H:%M:%S格式
    mid_time = pd.to_datetime(last_data_time) - pd.Timedelta(days=1)
    mid_time = mid_time.strftime('%Y-%m-%d %H:%M:%S')
    print(mid_time)

    end_time = pd.to_datetime(mid_time) + pd.Timedelta(minutes=5)
    end_time = end_time.strftime('%Y-%m-%d %H:%M:%S')
    print(end_time)

    # 划分训练集和测试集
    train_df = df[(df.index < mid_time) & (df.index > start_time)].values
    test_df = df[(df.index >= mid_time) & (df.index < end_time)].values

    print(train_df.shape)
    print(test_df.shape)

    sc = MinMaxScaler(feature_range=(0, 1))
    train_df_scaled = sc.fit_transform(train_df)

    x_train = []
    y_train = []
    for i in range(steps, train_df_scaled.shape[0]):
        x_train.append(train_df_scaled[i-steps:i, 0])
        y_train.append(train_df_scaled[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # 构建只有一层的LSTM模型
    model = Sequential()
    model.add(LSTM(units=128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))

    model.add(LSTM(units=128))
    model.add(Dropout(0.2))

    model.add(Dense(units=1))

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

    model.compile(optimizer='rmsprop', loss='mse')

    model.fit(x_train, y_train, epochs=20, batch_size=32)

    # end_time_list = []
    # end_time1 = pd.to_datetime(end_time) + pd.Timedelta(minutes=5)
    # end_time1 = end_time1.strftime('%Y-%m-%d %H:%M:%S')
    # end_time_list.append(end_time1)
    # end_time1 = pd.to_datetime(end_time) + pd.Timedelta(minutes=15)
    # end_time1 = end_time1.strftime('%Y-%m-%d %H:%M:%S')
    # end_time_list.append(end_time1)
    # end_time1 = pd.to_datetime(end_time) + pd.Timedelta(minutes=30)
    # end_time1 = end_time1.strftime('%Y-%m-%d %H:%M:%S')
    # end_time_list.append(end_time1)
    # end_time1 = pd.to_datetime(end_time) + pd.Timedelta(hours=1)
    # end_time1 = end_time1.strftime('%Y-%m-%d %H:%M:%S')
    # end_time_list.append(end_time1)
    # end_time1 = pd.to_datetime(end_time) + pd.Timedelta(hours=3)
    # end_time1 = end_time1.strftime('%Y-%m-%d %H:%M:%S')
    # end_time_list.append(end_time1)
    # end_time1 = pd.to_datetime(end_time) + pd.Timedelta(hours=6)
    # end_time1 = end_time1.strftime('%Y-%m-%d %H:%M:%S')
    # end_time_list.append(end_time1)
    # end_time1 = pd.to_datetime(end_time) + pd.Timedelta(hours=12)
    # end_time1 = end_time1.strftime('%Y-%m-%d %H:%M:%S')
    # end_time_list.append(end_time1)
    # end_time1 = pd.to_datetime(end_time) + pd.Timedelta(days=1)
    # end_time1 = end_time1.strftime('%Y-%m-%d %H:%M:%S')
    # end_time_list.append(end_time1)
    #
    # for end_time1 in end_time_list:
    #     df1 = df.tail(steps)
    #     test_df1 = df[(df.index >= mid_time) & (df.index < end_time1)]
    #     inputs = pd.concat([df1, test_df1], axis=0).values
    #
    #     inputs = inputs.reshape(-1, 1)
    #     inputs = sc.transform(inputs)
    #     x_test = []
    #     for i in range(steps, inputs.shape[0]):
    #         x_test.append(inputs[i - steps:i, 0])
    #     x_test = np.array(x_test)
    #
    #     x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    #
    #     predict_test = model.predict(x_test)  # 预测
    #     predict_test = sc.inverse_transform(predict_test)  # 反归一化
    #
    #     # 将训练数据，测试数据，预测数据呈现在一张图上
    #     plt.plot(test_df1.values, label='test')
    #     plt.plot(predict_test, label='predict')
    #     # 添加标题
    #     plt.title("lstm(steps:" + str(steps) + ")")
    #     plt.legend()
    #     plt.show()

    # 获取train_df的最后steps个数据，用于预测
    df1 = df.tail(steps)
    test_df1 = df[(df.index >= mid_time) & (df.index < end_time)]
    inputs = pd.concat([df1, test_df1], axis=0).values

    inputs = inputs.reshape(-1, 1)
    inputs = sc.transform(inputs)
    x_test = []
    for i in range(steps, inputs.shape[0]):
        x_test.append(inputs[i - steps:i, 0])
    x_test = np.array(x_test)

    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predict_test = model.predict(x_test)  # 预测
    predict_test = sc.inverse_transform(predict_test)  # 反归一化

    # 将训练数据，测试数据，预测数据呈现在一张图上
    plt.plot(test_df, label='test')
    plt.plot(predict_test, label='predict')
    # 添加标题
    plt.title("lstm(steps:" + str(steps) + ")")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    lstm_model_test("cs4.csv", 14)
