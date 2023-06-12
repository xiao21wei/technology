# 构建prophet模型，进行多变量时序预测,Ng_GenPCal.csv文件中的数据为：time,Ng,GenPCal,需要预测的值为GenPCal
import itertools

import pandas as pd
from matplotlib import pyplot as plt
from prophet import Prophet


def prophet_model_pro_test():
    df = pd.read_csv('Ng_GenPCal.csv', parse_dates=['time'])
    df = df.rename(columns={'time': 'ds', 'GenPCal': 'y'})
    print(df.shape)

    test_split = round(len(df) * 0.20)
    df_for_training = df[:-test_split]
    df_for_testing = df[-test_split:]
    print(df_for_training.shape)
    print(df_for_testing.shape)

    # param_grid = {
    #     'n_changepoints': [11],
    #     'changepoint_range': [0.3],
    #     'seasonality_mode': ['additive'],
    #     'seasonality_prior_scale': [0.05],
    #     'interval_width': [0.8, 0.85, 0.9, 0.95]
    # }
    #
    # all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    # rmses = []  # 用于存储各个参数集对应的RMSE误差
    #
    # # Use cross validation to evaluate all parameters
    # for params in all_params:
    #     m = Prophet(**params).add_regressor('Ng')  # Fit model with given params
    #     m.fit(df_for_training)  # Fit model with given params
    #     df_cv = m.predict(df_for_testing)  # Make predictions
    #     df_p = df_cv[['ds', 'yhat']].join(df_for_testing[['ds', 'y']].set_index('ds'), on='ds')  # Predictions and test data
    #     df_p.dropna(inplace=True)
    #     rmses.append((params, (df_p['y'] - df_p['yhat']).apply(lambda x: x ** 2).mean() ** 0.5))
    #
    # # Find the best parameters
    # best_params = all_params[rmses.index(min(rmses, key=lambda x: x[1]))]
    # print(best_params)

    param = {
        'n_changepoints': 11,
        'changepoint_range': 0.3,
        'seasonality_mode': 'additive',
        'seasonality_prior_scale': 0.05,
        'interval_width': 0.8
    }

    # 构建模型
    model = Prophet(**param)
    model.add_regressor('Ng')
    model.fit(df_for_training)

    # # 构建模型
    # model = Prophet(**best_params)
    # model.add_regressor('Ng')
    # model.fit(df_for_training)
    # 预测
    forecast = model.predict(df_for_testing)

    # 预测结果
    model.plot(forecast)
    plt.show()

    # 将测试数据和预测数据呈现在一张图上
    plt.plot(df_for_testing['ds'], df_for_testing['y'], label='test')
    plt.plot(forecast['ds'], forecast['yhat'], label='forecast')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    prophet_model_pro_test()
