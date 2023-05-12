from binance.client import Client
import ta
import mplfinance as mpf
import pandas as pd
import matplotlib.pyplot as plt
import requests
from datetime import datetime,timedelta
import time
import json



class BinanceFunction:
    def __init__(self, api_key, api_secret):
        self.client = Client(api_key, api_secret)
    def plot_candlestick(self,symbol, interval):
        # Get the Klines data
        klines = self.client.get_historical_klines(symbol, interval, "1 day ago UTC")

        # Create a Pandas DataFrame
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                           'close_time', 'quote_asset_volume', 'number_of_trades',
                                           'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])

        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        # Set the index to timestamp
        df.set_index('timestamp', inplace=True)

        # Convert columns to float
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)

        # 绘制K线图
        mpf.plot(df, type='candle', style='charles', title=symbol, ylabel='Price ($)', volume=True)
    def plot_moving_average(self,symbol, interval):
        # Get the Klines data
        klines = self.client.get_historical_klines(symbol, interval, "1 week ago UTC")

        # Create a Pandas DataFrame
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                           'close_time', 'quote_asset_volume', 'number_of_trades',
                                           'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])

        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        # Set the index to timestamp
        df.set_index('timestamp', inplace=True)

        # Convert columns to float
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)

        # 计算7天和30天移动平均线
        df['MA7'] = df['close'].rolling(window=7).mean()
        df['MA30'] = df['close'].rolling(window=30).mean()

        # 绘制移动平均线图
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['close'], label='Price')
        plt.plot(df.index, df['MA7'], label='MA7')
        plt.plot(df.index, df['MA30'], label='MA30')
        plt.title(symbol + ' Moving Average')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.show()
    def plot_rsi(self,symbol, interval):
        # Get the Klines data
        klines = self.client.get_historical_klines(symbol, interval, "1 week ago UTC")

        # Create a Pandas DataFrame
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                           'close_time', 'quote_asset_volume', 'number_of_trades',
                                           'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])

        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        # Set the index to timestamp
        df.set_index('timestamp', inplace=True)

        # Convert columns to float
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)

        # 计算RSI指标
        df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()

        # 绘制RSI图
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['RSI'])
        plt.title(symbol + ' Relative Strength Index')
        plt.xlabel('Date')
        plt.ylabel('RSI')
        plt.show()
    def order(self, symbol, side, quantity):
        order = self.client.create_order(
            symbol=symbol,
            side=side,
            type=Client.ORDER_TYPE_MARKET,
            quantity=quantity)
        return order
    def get_binance_server_time(self):
        url = "https://api.binance.com/api/v3/time"
        response = requests.get(url)
        data = response.json()
        server_time = data["serverTime"]
        timestamp_s = server_time / 1000
        dt = datetime.fromtimestamp(timestamp_s)
        return dt
    def timeDif(self):
        url = "https://api.binance.com/api/v1/time"
        t = time.time() * 1000
        r = requests.get(url)
        result = json.loads(r.content)
        print(int(t) - result["serverTime"])
    def get_all_tickers(self):
        """获取Binance上所有虚拟货币的行情信息"""
        tickers = self.client.get_all_tickers()
        return tickers
    def count_gainers_and_losers(self):
        """统计所有虚拟货币的涨跌情况"""
        tickers = self.get_all_tickers()
        num_gainers = 0
        num_losers = 0
        for ticker in tickers:
            price_change = float(ticker['priceChange'])
            if price_change > 0:
                num_gainers += 1
            else:
                num_losers += 1
        return num_gainers, num_losers
    def change_percent(self,coin,delta):
        now = datetime.now()
        # 计算 n 天前的时间
        delta = timedelta(days= delta)
        start_time = now - delta

        # 将时间转换为毫秒级时间戳
        start_time_ms = int(start_time.timestamp() * 1000)

        # 获取某个货币对过去 n 天的 kline 数据
        klines = self.client.get_historical_klines(coin, Client.KLINE_INTERVAL_1DAY, start_time_ms)

        # 计算涨幅
        start_price = float(klines[0][1])
        end_price = float(klines[-1][4])
        price_change = (end_price - start_price) / start_price * 100

        return price_change
    def transfer_to_spot(self, asset, amount):
        """将钱包中的币转到现货账户"""
        transfer_from = 'SPOT'  # 转出账户类型
        transfer_to = 'SPOT'  # 转入账户类型
        transfer_type = 'MAIN_C2C'
        resp = self.client.transfer_dust(asset=asset, amount=amount, from_account=transfer_from, to_account=transfer_to,
                             transfer_type=transfer_type)
        print(f"Transfer from {transfer_from} to {transfer_to}: {resp}")
    def sell(self, symbol: str, quantity: float):
        current_price = self.client.get_symbol_ticker(symbol=symbol)
        current_price = float(current_price['price'])
        order = self.client.create_order(
            symbol=symbol,
            side=Client.SIDE_SELL,
            type=Client.ORDER_TYPE_MARKET,
            quantity=quantity
        )
        return order,current_price
    def buy(self, symbol: str, quote_quantity: float):
        # 获取当前市场价格
        current_price = self.client.get_symbol_ticker(symbol=symbol)
        current_price = float(current_price['price'])

        # 计算购买的基础货币数量
        base_quantity = quote_quantity / current_price

        # 获取交易对的最小交易单位信息
        symbol_info = self.client.get_symbol_info(symbol)
        step_size = float([x['stepSize'] for x in symbol_info['filters'] if x['filterType'] == 'LOT_SIZE'][0])

        # 根据最小交易单位对购买数量进行取整
        base_quantity = int(base_quantity / step_size) * step_size

        order = self.client.create_order(
            symbol=symbol,
            side=Client.SIDE_BUY,
            type=Client.ORDER_TYPE_MARKET,
            quantity=base_quantity
        )

        # 将买入价格添加到订单信息中
        order['buy_price'] = current_price

        return order
    def get_symbol_info(self,symbol):
        symbol_info = self.client.get_symbol_info(symbol)
        return symbol_info
    def get_min_notional(self,symbol):
        symbol_info = self.client.get_symbol_info(symbol)
        min_notional = None
        for filter in symbol_info['filters']:
            if filter['filterType'] == 'NOTIONAL':
                min_notional = float(filter['NOTIONAL'])
                break
        return min_notional
    def get_klines_data(self, symbol: str, interval: str, start_time: str):
        klines_data = self.client.futures_klines(
            symbol=symbol,
            interval=interval,
            startTime=start_time,
        )
        df = pd.DataFrame(klines_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                                'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
                                                'taker_buy_quote_asset_volume', 'ignore'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    def get_symbols(self):
        account_info = self.client.get_account()
        balances = account_info['balances']
        symbols = [b['asset'] for b in balances if float(b['free']) > 0]
        # 对于每个交易对，获取您的余额
        for symbol in symbols:
            balance = self.client.get_asset_balance(asset=symbol)
            print(symbol, balance['free'])
        return symbols
