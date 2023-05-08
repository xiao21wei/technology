import pandas as pd
from matplotlib import pyplot as plt
from prophet import Prophet


def prophet_model_train(df):
    # Create the Prophet model
    model = Prophet()
    # Fit the model to the data
    model.fit(df)
    return model


def prophet_model_predict(model, future):
    # Make predictions
    forecast = model.predict(future)
    return forecast


def prophet_model_test(csv_file, value):
    # Load the data
    df = pd.read_csv(csv_file, parse_dates=['trendTime'])
    # Rename the columns
    df = df.rename(columns={'trendTime': 'ds', value: 'y'})
    last_data_time = df['ds'].iloc[-1]
    print(last_data_time)
    # last_data_time为%Y-%m-%d %H:%M:%S格式,计算半天前的时间
    last_data_time = pd.to_datetime(last_data_time) - pd.Timedelta(hours=12)
    last_data_time = last_data_time.strftime('%Y-%m-%d %H:%M:%S')
    print(last_data_time)
    # 划分训练集和测试集
    df_train = df[df['ds'] < last_data_time]
    df_test = df[df['ds'] >= last_data_time]

    print(df_train.shape)
    print(df_test.shape)

    # Create the Prophet model and fit on the data
    model = prophet_model_train(df_train)
    # Create a future dataframe with the same column names
    future = pd.DataFrame(df_test['ds'])
    # Make predictions for the future dates
    forecast = prophet_model_predict(model, future)
    model.plot(forecast)

    # 将测试数据和预测数据呈现在一张图上
    plt.plot(df_test['ds'], df_test['y'], label='test')
    plt.plot(forecast['ds'], forecast['yhat'], label='forecast')
    plt.legend()
    plt.show()

    plt.plot(df_train['ds'], df_train['y'], label='train')
    plt.plot(df_test['ds'], df_test['y'], label='test')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    prophet_model_test('1Aa.csv', 'three')
