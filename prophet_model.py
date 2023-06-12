import itertools

import pandas as pd
from matplotlib import pyplot as plt
from prophet import Prophet


def prophet_model_test(csv_file, value):
    # Load the data
    df = pd.read_csv(csv_file, parse_dates=['time'])
    # Rename the columns
    df = df.rename(columns={'time': 'ds', value: 'y'})
    last_data_time = df['ds'].iloc[-1]

    start_time = pd.to_datetime(last_data_time) - pd.Timedelta(days=8)
    start_time = start_time.strftime('%Y-%m-%d %H:%M:%S')
    start_time = '2021-11-02 11:49:38'
    print(start_time)

    # last_data_time为%Y-%m-%d %H:%M:%S格式
    mid_time = pd.to_datetime(last_data_time) - pd.Timedelta(days=1)
    mid_time = mid_time.strftime('%Y-%m-%d %H:%M:%S')
    mid_time = '2021-11-04 04:22:01'
    print(mid_time)

    end_time = pd.to_datetime(mid_time) + pd.Timedelta(days=1)
    end_time = end_time.strftime('%Y-%m-%d %H:%M:%S')
    end_time = '2021-11-04 15:02:32'
    print(end_time)

    # 划分训练集和测试集
    df_train = df[(df['ds'] < mid_time) & (df['ds'] > start_time)]
    df_test = df[(df['ds'] >= mid_time) & (df['ds'] < end_time)]

    print(df_train.shape)
    print(df_test.shape)

    # param_grid = {
    #     'changepoint_range': [i / 10 for i in range(3, 10)],
    #     'seasonality_mode': ['additive', 'multiplicative'],
    #     'seasonality_prior_scale': [0.05, 0.1, 0.5, 1, 5, 10, 15],
    # }
    #
    # all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    # rmses = []  # 用于存储各个参数集对应的RMSE误差
    #
    # # Use cross validation to evaluate all parameters
    # for params in all_params:
    #     m = Prophet(**params).fit(df_train)  # Fit model with given params
    #     df_cv = m.predict(df_test)  # Make predictions
    #     df_p = df_cv[['ds', 'yhat']].join(df_test[['ds', 'y']].set_index('ds'), on='ds')  # Predictions and test data
    #     df_p.dropna(inplace=True)
    #     rmses.append((params, (df_p['y'] - df_p['yhat']).apply(lambda x: x ** 2).mean() ** 0.5))
    #
    # # Find the best parameters
    # best_params = all_params[rmses.index(min(rmses, key=lambda x: x[1]))]
    # print(best_params)

    # best_params = {
    #     'changepoint_range': 0.8,
    #     'seasonality_mode': 'multiplicative',  # 'multiplicative
    #     'seasonality_prior_scale': 10,
    # }

    # Use the best params to fit the model
    # m = Prophet(**best_params).fit(df_train)
    m = Prophet().fit(df_train)
    forecast = m.predict(df_test)

    m.plot(forecast)
    plt.show()

    # 将测试数据和预测数据呈现在一张图上
    plt.plot(df_test['ds'], df_test['y'], label='test')
    plt.plot(forecast['ds'], forecast['yhat'], label='forecast')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    prophet_model_test('Ng.csv', 'Ng')
