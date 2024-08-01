## 航空券価格予測

このリポジトリは、旅行予約サイト「トラベルコ」（https://www.tour.ne.jp/）から航空券価格を取得し、その価格推移を予測し、価格が下がる確率を計算するPythonスクリプト群である。

### 機能

* **データ収集:** Seleniumを用いてトラベルコから航空券価格データを取得する。
* **データ前処理:** pandasを用いてデータを整形し、曜日情報をOne-hotエンコーディングで表現する。
* **データリサンプリング:** 30分ごとの価格データにリサンプリングし、欠損値は線形補完する。
* **価格予測:** TensorFlow/Kerasを用いてLSTMモデルを構築し、過去の価格データに基づいて将来の価格を予測する。
* **価格低下確率計算:** 予測価格と過去の価格変動に基づき、価格が下がる確率を計算する。
* **結果の可視化:** matplotlibを用いて実際の価格と予測価格をグラフで表示する。

### 使用ライブラリ

* **Selenium:** Webブラウザの自動操作 ([https://www.selenium.dev/](https://www.selenium.dev/))
* **pandas:** データ分析・操作 ([https://pandas.pydata.org/](https://pandas.pydata.org/))
* **TensorFlow/Keras:** 機械学習 ([https://www.tensorflow.org/](https://www.tensorflow.org/))
* **matplotlib:** データ可視化 ([https://matplotlib.org/](https://matplotlib.org/))
* **NumPy:** 数値計算 ([https://numpy.org/](https://numpy.org/))
* **SciPy:** 科学技術計算 ([https://scipy.org/](https://scipy.org/))
* **Optuna:** ハイパーパラメータ自動最適化 ([https://optuna.org/](https://optuna.org/))

### ファイル説明

* `data_create_csv.py`: トラベルコから航空券価格データを取得し、CSVファイルに保存する。
* `data.csv`: `data_create_csv.py` によって収集されたデータである。
* `resample.py`: `data.csv` を読み込み、データをリサンプリングして `data_resample.csv` に保存する。
* `data_resample.csv`: `resample.py` によってリサンプリングされたデータである。
* `best_param_search.py`: Optunaを使用してLSTMモデルのハイパーパラメータを最適化する。
* `prediction.py`: LSTMモデルを用いて航空券価格を予測する。
* `evaluation.py`: 予測結果を評価し、価格が下がる確率を計算して表示する。

### 実行方法

1. 必要なライブラリをインストールする。
2. `data_create_csv.py` を実行してトラベルコから航空券価格データを取得する。
3. `resample.py` を実行してデータをリサンプリングする。
4. `best_param_search.py` を実行してLSTMモデルのハイパーパラメータを最適化する (任意)。
5. `evaluation.py` を実行して予測結果を表示する。 