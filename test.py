import pandas as pd
from sklearn.preprocessing import MinMaxScaler

if __name__ == '__main__':
    df = pd.read_csv('Ng_GenPCal.csv', parse_dates=['time'], index_col=[0])

    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(df)

    print(df_scaled.shape)
    print(df_scaled[0:4, 0])
