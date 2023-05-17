import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor


def randomForest_model_train(x, y):
    # Create the random forest model
    model = RandomForestRegressor()
    # Fit the model to the data
    model.fit(x, y)
    return model


def randomForest_model_predict(model, future):
    # Make predictions
    forecast = model.predict(future)
    return forecast


def randomForest_model_test(csv_file, value, n):
    # Load the data
    df = pd.read_csv(csv_file, parse_dates=['trendTime'])
    # Rename the columns
    df = df.rename(columns={'trendTime': 'ds', value: 'y'})
    last_data_time = df['ds'].iloc[-1]

    start_time = pd.to_datetime(last_data_time) - pd.Timedelta(days=8)
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
    data = df.copy()

    for i in range(1, n+1):
        data['ypre_' + str(i)] = data['y'].shift(i)
    data = data[['ds'] + ['ypre_' + str(i) for i in range(n, 0, -1)] + ['y']]  # 选择特征

    x_train = data[(data['ds'] < mid_time) & (data['ds'] > start_time)].dropna()[['ypre_' + str(i) for i in range(n, 0, -1)]]
    y_train = data[(data['ds'] < mid_time) & (data['ds'] > start_time)].dropna()['y']
    x_test = data[(data['ds'] >= mid_time) & (data['ds'] < end_time)].dropna()[['ypre_' + str(i) for i in range(n, 0, -1)]]
    y_test = data[(data['ds'] >= mid_time) & (data['ds'] < end_time)].dropna()['y']

    # Create the random forest model
    model = randomForest_model_train(x_train, y_train)
    # Make predictions
    forecast = randomForest_model_predict(model, x_test)
    # # 将训练数据，测试数据，预测数据呈现在一张图上
    # plt.plot(x_train.index, y_train, label='train')
    # plt.plot(x_test.index, y_test, label='test')
    # plt.plot(x_test.index, forecast, label='forecast')
    # plt.legend()
    # plt.show()
    # 将测试数据和预测数据呈现在一张图上
    plt.plot(x_test.index, y_test, label='test')
    plt.plot(x_test.index, forecast, label='forecast')
    plt.title("randomForest(window_length:"+str(n)+")")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # for i in range(5, 31):
    #     randomForest_model_test('cs4.csv', 'three', i)
    randomForest_model_test('cs4.csv', 'three', 10)
