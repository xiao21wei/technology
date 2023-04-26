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
    df = pd.read_csv(csv_file)
    # Rename the columns
    df = df.rename(columns={'trendTime': 'ds', value: 'y'})
    last_data_time = df['ds'].iloc[-1]
    print(last_data_time)
    # last_data_time为%Y-%m-%d %H:%M:%S格式,计算一天前的时间
    last_data_time = pd.to_datetime(last_data_time) - pd.Timedelta(days=1)
    last_data_time = last_data_time.strftime('%Y-%m-%d %H:%M:%S')
    print(last_data_time)
    # 划分训练集和测试集
    train_df = df[df['ds'] < last_data_time]
    test_df = df[df['ds'] >= last_data_time]
    # Create the Prophet model
    model = prophet_model_train(train_df)
    # Create the future dataframe
    future = test_df[['ds']]
    # Make predictions
    forecast = prophet_model_predict(model, future)
    # Plot the predictions
    plt.plot(test_df['ds'], test_df['y'])
    plt.plot(future['ds'], forecast['yhat'])
    plt.show()


if __name__ == '__main__':
    prophet_model_test('35cd.csv', 'all')
