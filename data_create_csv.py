#!/usr/bin/env python3
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
from datetime import datetime, timedelta

# ChromeDriverのパス
chromedriver_path = '省略'

def get_element_with_retry(driver, by, value, retries=3):
    """
    指定された要素を再試行して取得する。

    Args:
        driver (webdriver.Chrome): WebDriverのインスタンス。
        by (By): 要素を見つけるための方法。
        value (str): 要素を見つけるための値。
        retries (int): 再試行の回数。

    Returns:
        WebElement: 見つけたWeb要素。

    Raises:
        Exception: 再試行の後も要素が見つからない場合に発生する。
    """
    for i in range(retries):
        try:
            return driver.find_element(by, value)
        except Exception as e:
            if i == retries - 1:
                raise e
            time.sleep(1)

def initialize_driver(chromedriver_path):
    """
    Chrome WebDriverを初期化する。

    Args:
        chromedriver_path (str): ChromeDriver実行ファイルのパス。

    Returns:
        webdriver.Chrome: 初期化されたWebDriverのインスタンス。
    """
    service = Service(chromedriver_path)
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    return webdriver.Chrome(service=service, options=options)

def fetch_flight_data(driver, departure_date, return_date):
    """
    指定されたURLからフライトデータを取得する。

    Args:
        driver (webdriver.Chrome): WebDriverのインスタンス。
        departure_date (str): 出発日（'YYYYMMDD'形式）。
        return_date (str): 帰国日（'YYYYMMDD'形式）。

    Returns:
        tuple: フォーマットされた出発日、帰国日、及び価格を含むタプル。

    Raises:
        Exception: データの取得または解析にエラーが発生した場合。
    """
    url = f'https://www.tour.ne.jp/w_air/list/?slice_info=TYO-HNL%7CHNL-TYO#dpt_date={departure_date}%7C{return_date}'
    driver.get(url)
    
    # JavaScriptとAjaxのロードが完了するまで待つ
    time.sleep(30)
    wait = WebDriverWait(driver, 30)
    wait.until(lambda driver: driver.execute_script('return document.readyState') == 'complete')

    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(3)

    wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'span.flight-summary-total-price')))

    flight = driver.find_element(By.CSS_SELECTOR, '.search-result-item')

    price_element = get_element_with_retry(flight, By.CSS_SELECTOR, 'span.flight-summary-total-price')

    price_text = price_element.text.replace('¥', '').replace(',', '').replace('円', '').strip()
    price = int(price_text)

    # URLから出発日と帰国日を取得
    departure_date_str = url.split('#dpt_date=')[1].split('%')[0]
    departure_date_obj = datetime.strptime(departure_date_str, '%Y%m%d')
    departure_date_formatted = departure_date_obj.strftime('%Y年%m月%d日')
    return_date_str = url.split('#dpt_date=')[1].split('%7C')[1]
    return_date_obj = datetime.strptime(return_date_str, '%Y%m%d')
    return_date_formatted = return_date_obj.strftime('%Y年%m月%d日')

    return departure_date_formatted, return_date_formatted, price

def main():
    """
    メインの処理を実行する。フライト情報を取得し、CSVファイルに保存する。
    """
    driver = initialize_driver(chromedriver_path)
    
    flight_data = []
    departure_dates = []
    return_dates = []

    for date_offset in range(8):  # 8日間ループ
        departure_date = (datetime(2024, 8, 2) + timedelta(days=date_offset)).strftime('%Y%m%d')
        return_date = (datetime(2024, 8, 4) + timedelta(days=date_offset)).strftime('%Y%m%d')

        try:
            dep_date, ret_date, price = fetch_flight_data(driver, departure_date, return_date)
            departure_dates.append(dep_date)
            return_dates.append(ret_date)
            flight_data.append([price])
        except Exception as e:
            print(f"エラー発生: {e}")
            continue

        print(f"取得した便情報: {flight_data[-1]}")

    driver.quit()

    if not flight_data:
        print("便の情報が取得できませんでした。")
    else:
        print(f"取得した便情報: {flight_data}")

        # 現在の日時を取得
        obtain_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # データをDataFrameに記録
        df_new = pd.DataFrame({
            '取得日時': [obtain_date] * len(flight_data),
            '便1 出発日': departure_dates,
            '便1 帰国日': return_dates,
            '便1 価格': [data[0] for data in flight_data]
        })

        # CSVファイルにデータを保存
        df_new.to_csv('data.csv', index=False, encoding='utf-8-sig')
        print('データをCSVファイルに保存しました。')

if __name__ == "__main__":
    main()
