# 构建randomForest模型，进行多变量时序预测,Ng_GenPCal.csv文件中的数据为：time,Ng,GenPCal,需要预测的值为GenPCal
import itertools

import numpy as np
import pandas as pd
from keras.losses import mean_squared_error
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor

import os

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def randomForest_model_pro_test():
    # Load the data
    df = pd.read_csv('Ng_GenPCal.csv', parse_dates=['time'])
    # Rename the columns
    df = df.rename(columns={'time': 'ds', 'GenPCal': 'y'})

    # 划分训练集和测试集
    test_split = round(len(df) * 0.20)
    train_split = len(df) - test_split

    # 将Ng的值作为特征，将GenPCal的值作为标签，构建训练集和测试集
    x_train = df.iloc[:train_split, 1:2]
    y_train = df.iloc[:train_split, 2:3]
    x_test = df.iloc[train_split:, 1:2]
    y_test = df.iloc[train_split:, 2:3]
    y_train = np.array(y_train).reshape(-1)

    # 构建模型，并进行调参
    param_grid = {
        'n_estimators': [139],
        'max_depth': [4],
        'min_samples_split': [5],
        'min_samples_leaf': [3],
        'max_features': ['sqrt', 'log2']
    }

    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    rmses = []
    for params in all_params:
        model = RandomForestRegressor(**params)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        # 计算rmse的均值
        rmse = np.mean(rmse)
        print(params, rmse)
        rmses.append(rmse)

    # # 画出rmse的变化曲线，横坐标为参数中的n_estimators，纵坐标为rmse
    # plt.figure(figsize=(20, 10))
    # plt.plot([i['min_samples_leaf'] for i in all_params], rmses)
    # plt.show()

    # 找出最优参数
    best_params = all_params[np.argmin(rmses)]
    print(best_params)

    # 构建模型
    model = RandomForestRegressor(**best_params)

    # model = RandomForestRegressor(n_estimators=100, max_depth=7, min_samples_split=2, min_samples_leaf=1)

    # 训练模型
    model.fit(x_train, y_train)

    # 预测
    forecast = model.predict(x_test)

    # 将测试数据和预测数据呈现在一张图上
    plt.plot(x_test.index, y_test, label='test')
    plt.plot(x_test.index, forecast, label='forecast')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    randomForest_model_pro_test()
