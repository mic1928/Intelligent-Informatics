import optuna
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, Bidirectional
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# データセットの準備
def create_direct_data(data, time_steps=1, predict_days=1):
    X, Y = [], []
    for i in range(len(data) - time_steps - predict_days + 1):
        X.append(data[i:(i + time_steps), :])  # すべての特徴量を含む
        Y.append(data[(i + time_steps):(i + time_steps + predict_days), 0])  # 価格のみ
    return np.array(X), np.array(Y)

# モデルの構築
def build_direct_model(input_shape, predict_days, lstm_units, learning_rate):
    input_layer = Input(shape=(input_shape[1], input_shape[2]), name='input_layer')
    
    # 双方向LSTM層を追加
    lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=False))(input_layer)
    output = Dense(predict_days)(lstm_out)
    
    model = Model(inputs=input_layer, outputs=output)
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

# データの読み込みと前処理
def preprocess_data(file_path):
    data = pd.read_csv(file_path, parse_dates=['取得日時'])
    data = data[['取得日時', '価格']].sort_values('取得日時')
    
    # 曜日情報の生成とエンコーディング
    data['曜日'] = data['取得日時'].dt.dayofweek
    onehot_encoder = OneHotEncoder(sparse=False)
    weekday_onehot = onehot_encoder.fit_transform(data['曜日'].values.reshape(-1, 1))
    
    # データのスケーリング（価格）
    scaler = MinMaxScaler(feature_range=(0, 1))
    data['価格'] = scaler.fit_transform(data[['価格']])
    
    # データセットの作成
    data_features = np.hstack((data[['価格']].values, weekday_onehot))
    return data_features, scaler

# Optunaによるハイパーパラメータ最適化
def objective(trial):
    file_path = 'data9_resample.csv'  # 適切なファイルパスに変更してください
    data_features, scaler = preprocess_data(file_path)
    
    time_steps = 200
    predict_days = 10
    
    # ハイパーパラメータのサンプリング
    lstm_units = trial.suggest_int('lstm_units', 50, 200)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    batch_size = trial.suggest_int('batch_size', 16, 128)
    epochs = trial.suggest_int('epochs', 20, 100)
    
    X, Y = create_direct_data(data_features, time_steps, predict_days)
    
    # データの分割
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    model = build_direct_model(X.shape, predict_days, lstm_units, learning_rate)
    
    model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, Y_val), verbose=0)
    
    # 検証損失を評価
    val_loss = model.evaluate(X_val, Y_val, verbose=0)
    return val_loss

if __name__ == "__main__":
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)
    
    print("Best trial:")
    trial = study.best_trial
    
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))