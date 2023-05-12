
# 'apiKey': 'gfvVH43y51jiEuzdFfNDwthxsmm06xLHCR24cGe1UVDsPVGXEfmFUeMLNOsqNFFj',
# 'secret': 'h2dS20ZdM4sOBiTyxl8xSPrqdTVBHmUEOiCt95ZnICKTZJNvBEFfTOqnj0EJDrkL',
import ta
import os
import time
import mplfinance as mpf
from binance.client import Client
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from BinanceSystem import BinanceFunction
from datetime import datetime, timedelta
# Binance API keys
api_key = 'gfvVH43y51jiEuzdFfNDwthxsmm06xLHCR24cGe1UVDsPVGXEfmFUeMLNOsqNFFj'
api_secret = 'h2dS20ZdM4sOBiTyxl8xSPrqdTVBHmUEOiCt95ZnICKTZJNvBEFfTOqnj0EJDrkL'

# Connect to Binance API
client = Client(api_key, api_secret)


bf = BinanceFunction(api_key, api_secret)


end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# 获取过去7天时间

print(end_time)
hh = 3
start_time = (datetime.now() - timedelta(days=hh)).strftime('%Y-%m-%d %H:%M:%S')
print(start_time)
# 获取BTC/USDT在过去7天每小时的K线数据
df = bf.get_klines_data(symbol='BTCUSDT', interval=Client.KLINE_INTERVAL_30MINUTE, start_time=start_time, end_time=end_time)

# 将数据写入csv文件
df.to_csv('test.csv')

# symbol_info =bf.get_min_notional('ETHUSDT')
# print("Symbol info:")
# print(symbol_info)
#
# exchange_info = client.get_exchange_info()
# symbols = exchange_info['symbols']
# for symbol in symbols:
#     if symbol['symbol'] == 'BTCUSDT':
#         filters = symbol['filters']
#         for f in filters:
#             if f['filterType'] == 'LOT_SIZE':
#                 step_size = f['stepSize']
#                 print(f"BTC/USDT的交易量步长为{step_size}")
#                 break
#         break


