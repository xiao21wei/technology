import itertools

import numpy as np
import pandas as pd
from keras.losses import mean_squared_error
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor

import os

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def randomForest_model_test(csv_file, value):
    # Load the data
    df = pd.read_csv(csv_file, parse_dates=['trendTime'])
    # Rename the columns
    df = df.rename(columns={'trendTime': 'ds', value: 'y'})
    last_data_time = df['ds'].iloc[-1]

    start_time = pd.to_datetime(last_data_time) - pd.Timedelta(days=1) - pd.Timedelta(days=7)
    start_time = start_time.strftime('%Y-%m-%d %H:%M:%S')

    # last_data_time为%Y-%m-%d %H:%M:%S格式
    mid_time = pd.to_datetime(last_data_time) - pd.Timedelta(days=1)
    mid_time = mid_time.strftime('%Y-%m-%d %H:%M:%S')

    end_time = pd.to_datetime(mid_time) + pd.Timedelta(days=1)
    end_time = end_time.strftime('%Y-%m-%d %H:%M:%S')

    # 划分训练集和测试集
    data = df.copy()
    n = 17

    for i in range(1, n + 1):
        data['ypre_' + str(i)] = data['y'].shift(i)
    data = data[['ds'] + ['ypre_' + str(i) for i in range(n, 0, -1)] + ['y']]  # 选择特征

    x_train = data[(data['ds'] < mid_time) & (data['ds'] > start_time)].dropna()[['ypre_' + str(i) for i in range(n, 0, -1)]]
    y_train = data[(data['ds'] < mid_time) & (data['ds'] > start_time)].dropna()['y']
    x_test = data[(data['ds'] >= mid_time) & (data['ds'] < end_time)].dropna()[['ypre_' + str(i) for i in range(n, 0, -1)]]
    y_test = data[(data['ds'] >= mid_time) & (data['ds'] < end_time)].dropna()['y']

    x_train = x_train.iloc[::1000, :]
    y_train = y_train.iloc[::1000]
    x_test = x_test.iloc[::1000, :]
    y_test = y_test.iloc[::1000]

    # # 构建模型，并进行调参
    # param_grid = {
    #     'n_estimators': [100, 200, 300, 400, 500],
    #     'max_depth': [3, 4, 5, 6, 7],
    #     'min_samples_split': [2, 3, 4, 5, 6],
    #     'min_samples_leaf': [1, 2, 3, 4, 5],
    #     'max_features': ['auto', 'sqrt', 'log2']
    # }
    # all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    # rmses = []
    #
    # for params in all_params:
    #     model = RandomForestRegressor(**params)
    #     model.fit(x_train, y_train)
    #     y_pred = model.predict(x_test)
    #     rmses.append(np.sqrt(mean_squared_error(y_test, y_pred)))
    #
    # # 打印最优参数
    # print('最优参数：', all_params[np.argmin(rmses)])

    # 选择最优模型
    # model = RandomForestRegressor(**all_params[np.argmin(rmses)])
    # model = RandomForestRegressor(n_estimators=500, max_depth=3, min_samples_split=4, min_samples_leaf=1, max_features='log2')

    # scorel = []
    # for i in range(145, 155):
    #     model = RandomForestRegressor(n_estimators=i)
    #     model.fit(x_train, y_train)
    #     score = model.score(x_test, y_test)
    #     scorel.append(score)
    # print(scorel)
    # plt.plot(range(145, 155), scorel)
    # plt.show()
    #
    # best_n = scorel.index(max(scorel)) + 145
    # print(best_n)
    #
    # model = RandomForestRegressor(n_estimators=best_n)

    # # 随机森林自动调参
    # param_grid = {
    #     'n_estimators': [153],
    #     'min_samples_split': np.arange(2, 22, 1),
    #     'min_samples_leaf': np.arange(1, 11, 1),
    #     'max_features': ['sqrt', 'log2']
    # }
    # model = RandomForestRegressor()
    # rf_grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=10)
    # rf_grid.fit(x_train, y_train)
    # print(rf_grid.best_params_)
    #
    # # 选择最优模型
    # model = RandomForestRegressor(**rf_grid.best_params_)

    model = RandomForestRegressor(n_estimators=153, min_samples_split=16, min_samples_leaf=10, max_features='sqrt')
    # 训练模型
    model.fit(x_train, y_train)
    # 预测
    forecast = model.predict(x_test)

    # 将测试数据和预测数据呈现在一张图上
    plt.plot(x_test.index, y_test, label='test')
    plt.plot(x_test.index, forecast, label='forecast')
    plt.title("randomForest(window_length:"+str(n)+")")
    plt.legend()
    plt.show()

    # # 比较不同特征窗口长度对模型的影响，选择最优窗口长度
    # scorel = []
    # for n in range(1, 30):
    #     data = df.copy()
    #
    #     for i in range(1, n + 1):
    #         data['ypre_' + str(i)] = data['y'].shift(i)
    #     data = data[['ds'] + ['ypre_' + str(i) for i in range(n, 0, -1)] + ['y']]  # 选择特征
    #
    #     x_train = data[(data['ds'] < mid_time) & (data['ds'] > start_time)].dropna()[['ypre_' + str(i) for i in range(n, 0, -1)]]
    #     y_train = data[(data['ds'] < mid_time) & (data['ds'] > start_time)].dropna()['y']
    #     x_test = data[(data['ds'] >= mid_time) & (data['ds'] < end_time)].dropna()[['ypre_' + str(i) for i in range(n, 0, -1)]]
    #     y_test = data[(data['ds'] >= mid_time) & (data['ds'] < end_time)].dropna()['y']
    #
    #     model = RandomForestRegressor(n_estimators=153, min_samples_split=16, min_samples_leaf=10, max_features='sqrt')
    #     # 训练模型
    #     model.fit(x_train, y_train)
    #     # 预测
    #     forecast = model.predict(x_test)
    #     score = model.score(x_test, y_test)
    #     scorel.append(score)
    # # 输出最大值对应的索引
    # print(scorel.index(max(scorel)) + 1)
    # plt.plot(range(1, 30), scorel)
    # plt.show()


if __name__ == '__main__':
    randomForest_model_test('cs4.csv', 'three')
