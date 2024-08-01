import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from prediction import main as prediction_main

def plot_predictions(data, predictions):
    """
    実際の価格データと予測価格をプロットする。

    Args:
        data (np.ndarray): 実際の価格データ。
        predictions (np.ndarray): 予測価格データ。
    """
    plt.plot(range(len(data)), data, label="Actual Prices")
    plt.plot(range(len(data), len(data) + len(predictions)), predictions, label="Predicted Prices", linestyle='--')
    
    # 実データの最後の点と予測データの最初の点を結ぶ線を追加
    plt.plot([len(data) - 1, len(data)], [data[-1], predictions[0]], 'k--', label="Connection Line")

    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.title("Price Predictions")
    plt.show()

def calculate_probability_of_decrease(last_actual_price, predictions, std_dev):
    """
    価格が下がる確率を計算する。

    Args:
        last_actual_price (float): 最後の実際の価格。
        predictions (np.ndarray): 予測価格データ。
        std_dev (float): 予測価格の標準偏差。

    Returns:
        float: 価格が下がる確率の最大値。
    """
    probabilities = []
    for pred in predictions:
        probability = norm.cdf(last_actual_price, loc=pred, scale=std_dev)
        probabilities.append(probability)
    
    # average_probability = np.mean(probabilities)
    max_probability = max(probabilities)
    return max_probability

def main():
    """
    メイン処理を実行する。予測結果の取得、グラフのプロット、価格が下がる確率の計算を行う。
    """
    # prediction.pyのmain関数を実行して予測結果を取得
    file_path = 'data_resample.csv'
    predictions = prediction_main(file_path)

    # data_resample.csvから最後の200データを読み込む
    data = pd.read_csv(file_path)
    last_200_data = data['価格'].values[-200:]

    print(f"現在の価格: {last_200_data[-1]}")
    
    # グラフのプロット
    plot_predictions(last_200_data, predictions)
    
    # 標準偏差を計算
    std_dev = np.std(last_200_data)
    
    # 価格が下がる確率の計算
    last_actual_price = last_200_data[-1]
    probability_of_decrease = calculate_probability_of_decrease(last_actual_price, predictions, std_dev)
    print(f"価格が下がる確率: {probability_of_decrease:.2f}")
    
    # 価格の評価
    if probability_of_decrease >= 0.8:
        print("価格はほぼ確実に下がるでしょう。今買うのは待ったほうが良いでしょう。")
    elif probability_of_decrease >= 0.6:
        print("価格は下がる可能性が高いです。購入を待つことを推奨します。")
    elif probability_of_decrease >= 0.4:
        print("価格は少し下がる可能性があります。様子見も検討しましょう。")
    elif probability_of_decrease >= 0.2:
        print("価格はあまり下がらない可能性があります。購入を検討しても良いかもしれません。")
    else:
        print("価格はほぼ確実に下がらないでしょう。今が買い時かもしれません。")

if __name__ == '__main__':
    main()
