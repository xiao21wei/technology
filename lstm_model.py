import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import LSTM, Dropout, Dense
from keras.wrappers.scikit_learn import KerasRegressor
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime


def lstm_model_test(csv_file, value):
    # Load the data
    custom_date_parser = lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
    df = pd.read_csv(csv_file, parse_dates=['trendTime'], date_parser=custom_date_parser, index_col='trendTime')
    # Rename the columns
    last_data_time = df.index[-1]
    print(last_data_time)
    # last_data_time为%Y-%m-%d %H:%M:%S格式,计算一天前的时间
    last_data_time = last_data_time - pd.Timedelta(days=1)
    print(last_data_time)
    # 划分训练集和测试集
    train_df = df[df.index < last_data_time]
    test_df = df[df.index >= last_data_time]

    # 归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_for_training_scaled = scaler.fit_transform(train_df)
    df_for_testing_scaled = scaler.fit_transform(test_df)

    # 划分训练集和测试集
    trainX, trainY = createXY(df_for_training_scaled, 30)
    testX, testY = createXY(df_for_testing_scaled, 30)

    grid_model = KerasRegressor(build_fn=build_model, verbose=1, validation_data=(testX, testY))
    parameters = {'batch_size': [16, 20], 'epochs': [8, 10], 'optimizer': ['adam', 'Adadelta']}

    grid = GridSearchCV(estimator=grid_model, param_grid=parameters, cv=2)

    grid_search = grid.fit(trainX, trainY)

    grid_search.best_params_ = {'batch_size': 16, 'epochs': 10, 'optimizer': 'adam'}
    my_model = grid_search.best_estimator_.model

    prediction = my_model.predict(testX)
    prediction_copies_array = np.repeat(prediction, 6, axis=-1)
    pred = scaler.inverse_transform(np.reshape(prediction_copies_array, (len(prediction), 6)))[:, 0]
    original_copies_array = np.repeat(testY, 6, axis=-1)
    original = scaler.inverse_transform(np.reshape(original_copies_array, (len(testY), 6)))[:, 0]

    plt.plot(original, color='red', label='Real')
    plt.plot(pred, color='blue', label='Predicted')
    plt.title('Prediction')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()


def createXY(dataset, n_past):
    dataX = []
    dataY = []
    for i in range(n_past, len(dataset)):
        dataX.append(dataset[i - n_past:i, 0:dataset.shape[1]])
        dataY.append(dataset[i, 0])
    return np.array(dataX), np.array(dataY)


def build_model(optimizer):
    grid_model = Sequential()
    grid_model.add(LSTM(50, return_sequences=True, input_shape=(30, 6)))
    grid_model.add(LSTM(50))
    grid_model.add(Dropout(0.2))
    grid_model.add(Dense(1))
    grid_model.compile(loss='mean_squared_error', optimizer=optimizer)
    return grid_model


if __name__ == '__main__':
    lstm_model_test('40cs.csv', 'all')
