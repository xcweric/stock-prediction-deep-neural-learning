import datetime
import pandas as pd
import yfinance as yf


def download_market_data(stock=['AMD'], startdate='2021-08-01', end=datetime.date.today()):
    start = pd.to_datetime(startdate)
    data = yf.download(stock, start=start, end=end)
    return data

def test():
    import tensorflow as tf

    # 检查 TensorFlow 是否能够检测到 GPU
    physical_devices = tf.config.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(physical_devices))

    # 测试 TensorFlow 是否能够使用 GPU
    if physical_devices:
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            print("TensorFlow successfully configured to use GPU.")
        except RuntimeError as e:
            print(e)
    else:
        print("TensorFlow did not detect any GPU.")

if __name__ == '__main__':
    # print(download_market_data(stock=['AMD'], startdate='2023-08-01'))
    test()
