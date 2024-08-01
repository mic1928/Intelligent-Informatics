# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense

# def load_data(file_path):
#     """
#     CSVファイルからデータを読み込む。

#     Args:
#         file_path (str): 読み込むCSVファイルのパス。

#     Returns:
#         pd.DataFrame: 読み込まれたデータフレーム。
#     """
#     data = pd.read_csv(file_path, parse_dates=['取得日時'])
#     data = data[['取得日時', '価格']].sort_values('取得日時')
#     return data

# def scale_data(data, scaler=None):
#     """
#     データをスケーリングする。

#     Args:
#         data (pd.DataFrame): スケーリングするデータ。
#         scaler (MinMaxScaler, optional): スケーラー。指定がない場合は新たに作成する。

#     Returns:
#         Tuple[pd.DataFrame, MinMaxScaler]: スケーリングされたデータとスケーラー。
#     """
#     if scaler is None:
#         scaler = MinMaxScaler(feature_range=(0, 1))
#         data['価格'] = scaler.fit_transform(data[['価格']])
#     else:
#         data['価格'] = scaler.transform(data[['価格']])
#     return data, scaler

# def create_lstm_data(data, time_steps=1):
#     """
#     LSTMモデルの入力データを作成する。

#     Args:
#         data (np.ndarray): 入力データ。
#         time_steps (int): 時系列のウィンドウサイズ。

#     Returns:
#         Tuple[np.ndarray, np.ndarray]: 入力データとラベル。
#     """
#     X, Y = [], []
#     for i in range(len(data) - time_steps):
#         X.append(data[i:(i + time_steps), 0])
#         Y.append(data[i + time_steps, 0])
#     return np.array(X), np.array(Y)

# def build_model(input_shape):
#     """
#     LSTMモデルを構築する。

#     Args:
#         input_shape (tuple): モデルの入力形状。

#     Returns:
#         tensorflow.keras.models.Sequential: 訓練可能なLSTMモデル。
#     """
#     model = Sequential()
#     model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
#     model.add(LSTM(50))
#     model.add(Dense(1))
#     model.compile(loss='mean_squared_error', optimizer='adam')
#     return model

# def predict_next_n_days(model, data, n_days):
#     """
#     次のn日間の価格を予測する。

#     Args:
#         model (tensorflow.keras.models.Sequential): 訓練済みモデル。
#         data (np.ndarray): 最後のn日間のデータ。
#         n_days (int): 予測するデータ数。

#     Returns:
#         List[float]: 予測された価格のリスト。
#     """
#     predictions = []
#     data = data.reshape(1, -1, 1)
#     for _ in range(n_days):
#         last_data = data[-1].reshape(1, data.shape[1], 1)
#         next_price = model.predict(last_data)
#         predictions.append(next_price[0, 0])
#         next_price = next_price.reshape(1, 1, 1)
#         data = np.append(data[:, 1:, :], next_price, axis=1)
#     return predictions

# def main(file_path):
#     """
#     データを処理し、次の10日間の価格を予測する。

#     Args:
#         file_path (str): 読み込むCSVファイルのパス。

#     Returns:
#         np.ndarray: 予測された価格。
#     """
#     # データの読み込み
#     data = load_data(file_path)
    
#     # データのスケーリング
#     data, scaler = scale_data(data)
    
#     # LSTMモデルのためのデータ準備
#     time_steps = 60  # 60個の過去データを使用
#     X, Y = create_lstm_data(data['価格'].values.reshape(-1, 1), time_steps)
#     X = X.reshape(X.shape[0], X.shape[1], 1)
    
#     # モデルの構築
#     model = build_model((X.shape[1], 1))
    
#     # モデルの訓練
#     model.fit(X, Y, epochs=20, batch_size=32, verbose=2)
    
#     # 次の10日間の予測
#     last_60_days = data['価格'].values[-time_steps:]
#     predictions = predict_next_n_days(model, last_60_days, 10)
    
#     # 予測値の逆スケーリング
#     predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
#     predictions = predictions.flatten()
    
#     # 予測結果の出力
#     print(f"次の10個の価格予測: {predictions}")

#     return predictions

# if __name__ == '__main__':
#     file_path = 'data5_resample.csv'
#     res = main(file_path)


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# データセットの準備
def create_direct_data(data, time_steps=1, predict_days=1):
    X, Y = [], []
    for i in range(len(data) - time_steps - predict_days + 1):
        X.append(data[i:(i + time_steps), 0])
        Y.append(data[(i + time_steps):(i + time_steps + predict_days), 0])
    return np.array(X), np.array(Y)

# モデルの構築
def build_direct_model(input_shape, predict_days):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50))
    model.add(Dense(predict_days))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# メインの流れ
def main(file_path):
    # データの読み込み
    data = pd.read_csv(file_path, parse_dates=['取得日時'])
    data = data[['取得日時', '価格']].sort_values('取得日時')
    
    # データのスケーリング
    scaler = MinMaxScaler(feature_range=(0, 1))
    data['価格'] = scaler.fit_transform(data[['価格']])
    
    # データセットの作成
    time_steps = 60  # 過去60日分を使用
    predict_days = 10  # 次の10日間を予測
    X, Y = create_direct_data(data['価格'].values.reshape(-1, 1), time_steps, predict_days)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    # モデルの構築と学習
    model = build_direct_model((X.shape[1], 1), predict_days)
    model.fit(X, Y, epochs=20, batch_size=32, verbose=2)
    
    # 予測
    last_60_days = data['価格'].values[-time_steps:].reshape(1, time_steps, 1)
    predictions = model.predict(last_60_days)
    predictions = scaler.inverse_transform(predictions)
    
    print(f"次の{predict_days}日間の価格予測: {predictions[0]}")
    return predictions[0]

# 実行
file_path = 'data5_resample.csv'
predictions = main(file_path)
