## 航空券価格予測

このリポジトリは、航空券価格の推移を予測し、価格が下がる確率を計算するPythonスクリプト群です。

### 機能

* **データ収集:** Seleniumを用いて旅行予約サイトから航空券価格データを取得します。
* **データ前処理:** pandasを用いてデータを整形し、曜日情報をOne-hotエンコーディングで表現します。
* **データリサンプリング:** 30分ごとの価格データにリサンプリングし、欠損値は線形補完します。
* **価格予測:** TensorFlow/Kerasを用いてLSTMモデルを構築し、過去の価格データに基づいて将来の価格を予測します。
* **価格低下確率計算:** 予測価格と過去の価格変動に基づき、価格が下がる確率を計算します。
* **結果の可視化:** matplotlibを用いて実際の価格と予測価格をグラフで表示します。

### 使用ライブラリ

* **Selenium:** Webブラウザの自動操作 ([https://www.selenium.dev/](https://www.selenium.dev/))
* **pandas:** データ分析・操作 ([https://pandas.pydata.org/](https://pandas.pydata.org/))
* **TensorFlow/Keras:** 機械学習 ([https://www.tensorflow.org/](https://www.tensorflow.org/))
* **matplotlib:** データ可視化 ([https://matplotlib.org/](https://matplotlib.org/))
* **NumPy:** 数値計算 ([https://numpy.org/](https://numpy.org/))
* **SciPy:** 科学技術計算 ([https://scipy.org/](https://scipy.org/))
* **Optuna:** ハイパーパラメータ自動最適化 ([https://optuna.org/](https://optuna.org/))

### ファイル説明

* `data_create_csv.py`: 旅行予約サイトから航空券価格データを取得し、CSVファイルに保存します。
* `data.csv`: `data_create_csv.py` によって収集されたデータ。
* `resample.py`: `data.csv` を読み込み、データをリサンプリングして `data_resample.csv` に保存します。
* `data_resample.csv`: `resample.py` によってリサンプリングされたデータ。
* `best_param_search.py`: Optunaを使用してLSTMモデルのハイパーパラメータを最適化します。
* `prediction.py`: LSTMモデルを用いて航空券価格を予測します。
* `evaluation.py`: 予測結果を評価し、価格が下がる確率を計算して表示します。

### 実行方法

1. 必要なライブラリをインストールします。
2. `data_create_csv.py` を実行して航空券価格データを取得します。
3. `resample.py` を実行してデータをリサンプリングします。
4. `best_param_search.py` を実行してLSTMモデルのハイパーパラメータを最適化します (任意)。
5. `evaluation.py` を実行して予測結果を表示します。
