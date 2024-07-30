import pandas as pd

def resample_and_interpolate(df):
    """
    '取得日時'をインデックスに設定し、30分ごとにリサンプリングし、線形補完を行います。

    Args:
        df (pd.DataFrame): 元のDataFrame。

    Returns:
        pd.DataFrame: リサンプリングおよび補完が適用されたDataFrame。
    """
    df['取得日時'] = pd.to_datetime(df['取得日時'])
    df = df.set_index('取得日時').resample('30T').first()
    df_resampled = df.interpolate(method='linear')
    df_resampled['価格'] = df_resampled['価格'].astype(int)
    return df_resampled.reset_index()[['取得日時', '価格']]

def main():
    """
    データを処理し、CSVファイルに保存します。
    """
    # CSVファイルを読み込む
    df = pd.read_csv('data6.csv')
    
    # 往路が2024年08月02日のデータのみ抽出
    df_filtered = df[df['往路'] == '2024年08月02日']
    
    # リサンプリングと線形補完を実行
    df_resampled = resample_and_interpolate(df_filtered)
    
    # CSVファイルに出力
    df_resampled.to_csv('data6_resample.csv', index=True)

    print(df_resampled)

if __name__ == "__main__":
    main()
