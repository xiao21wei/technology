import pandas as pd
from matplotlib import pyplot as plt
from prophet import Prophet


def prophet_model_test(csv_file, value):
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

    end_time = pd.to_datetime(mid_time) + pd.Timedelta(days=1)
    end_time = end_time.strftime('%Y-%m-%d %H:%M:%S')
    print(end_time)

    # 划分训练集和测试集
    df_train = df[(df['ds'] < mid_time) & (df['ds'] > start_time)]
    df_test = df[(df['ds'] >= mid_time) & (df['ds'] < end_time)]

    # 创建模型，并实现自动调参
    model = Prophet(changepoint_prior_scale=0.01, changepoint_range=0.9, n_changepoints=25, seasonality_mode='multiplicative')
    model.fit(df_train)

    forecast = model.predict(df_test)

    model.plot(forecast)
    plt.show()

    # 将测试数据和预测数据呈现在一张图上
    plt.plot(df_test['ds'], df_test['y'], label='test')
    plt.plot(forecast['ds'], forecast['yhat'], label='forecast')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    prophet_model_test('cs4.csv', 'three')
