import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, Bidirectional
from tensorflow.keras.optimizers import Adam

def create_direct_data(data, time_steps=1, predict_days=1):
    """
    時系列データからLSTM用の入力データと出力データを作成する。

    Args:
        data (np.ndarray): 時系列データ
        time_steps (int, optional): 入力データの時間ステップ数。デフォルトは1。
        predict_days (int, optional): 予測日数。デフォルトは1。

    Returns:
        tuple: 入力データ (X) と出力データ (Y)
    """
    X, Y = [], []
    for i in range(len(data) - time_steps - predict_days + 1):
        X.append(data[i:(i + time_steps), :])
        Y.append(data[(i + time_steps):(i + time_steps + predict_days), 0])
    return np.array(X), np.array(Y)

def build_direct_model(input_shape, predict_days, lstm_units, learning_rate):
    """
    LSTMモデルを構築する。

    Args:
        input_shape (tuple): 入力データの形状
        predict_days (int): 予測日数
        lstm_units (int): LSTM層のユニット数
        learning_rate (float): オプティマイザの学習率

    Returns:
        Model: コンパイル済みのLSTMモデル
    """
    input_layer = Input(shape=(input_shape[1], input_shape[2]), name='input_layer')
    lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=False))(input_layer)
    output = Dense(predict_days)(lstm_out)
    model = Model(inputs=input_layer, outputs=output)
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

def main(file_path):
    """
    データの読み込み、前処理、データセット作成、モデル構築・学習、および予測を行うメイン関数。

    Args:
        file_path (str): データファイルのパス

    Returns:
        np.ndarray: 指定された日数分の予測結果
    """
    # データの読み込みと前処理
    data = pd.read_csv(file_path, parse_dates=['取得日時'])
    data = data[['取得日時', '価格']].sort_values('取得日時')
    data['曜日'] = data['取得日時'].dt.dayofweek

    # 曜日のone-hotエンコーディング
    onehot_encoder = OneHotEncoder(sparse=False)
    weekday_onehot = onehot_encoder.fit_transform(data['曜日'].values.reshape(-1, 1))

    # 価格データの正規化
    scaler = MinMaxScaler(feature_range=(0, 1))
    data['価格'] = scaler.fit_transform(data[['価格']])

    # 価格とワンホットエンコードされた曜日データを結合
    data_features = np.hstack((data[['価格']].values, weekday_onehot))

    # データセットの作成
    time_steps = 200
    predict_days = 10
    X, Y = create_direct_data(data_features, time_steps, predict_days)

    # モデルの構築と学習
    lstm_units = 199
    learning_rate = 0.008189829991948023
    batch_size = 85
    epochs = 91
    model = build_direct_model(X.shape, predict_days, lstm_units, learning_rate)
    model.fit(X, Y, epochs=epochs, batch_size=batch_size, verbose=2)

    # 指定された日数分の予測を行う
    last_200_days_data = data_features[-time_steps:].reshape(1, time_steps, data_features.shape[1])
    predictions = model.predict(last_200_days_data)
    predictions = scaler.inverse_transform(predictions)

    print(f"次の{predict_days}日間の価格予測: {predictions[0]}")
    return predictions[0]

if __name__ == "__main__":
    file_path = 'data_resample.csv'
    predictions = main(file_path)



