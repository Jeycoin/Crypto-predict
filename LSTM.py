import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM,Dropout
import os
from binance.client import Client
from keras.models import load_model
from binance.exceptions import BinanceAPIException
import keras.backend as K

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))



modelspath = './models//'
class CryptoPricePrediction:
    def __init__(self, api_key, api_secret,symbol: str, interval: str, start_time: str, end_time: str, split_ratio: float, num_epochs: int):
        self.symbol = symbol
        self.interval = interval
        self.start_time = start_time
        self.end_time = end_time
        self.split_ratio = split_ratio
        self.num_epochs = num_epochs
        self.client = Client(api_key, api_secret)

    def get_crypto_prices(self):
        try:
            # 获取现货钱包中的虚拟货币余额
            account_info = self.client.get_account()
            balances = account_info['balances']

            # 获取交易对的价格
            ticker_prices = self.client.get_all_tickers()

            # 将价格存储为字典以便于查找
            price_dict = {ticker['symbol']: float(ticker['price']) for ticker in ticker_prices}

            # 计算各虚拟货币对应的USDT价值并存储在字典中
            crypto_prices = {}
            for balance in balances:
                asset = balance['asset']
                free_amount = float(balance['free'])
                if free_amount > 0:
                    if asset == 'USDT':
                        crypto_prices['USDT'] = free_amount
                    else:
                        usdt_symbol = f'{asset}USDT'
                        if usdt_symbol in price_dict:
                            usdt_value = free_amount * price_dict[usdt_symbol]
                            crypto_prices[asset] = usdt_value

            return crypto_prices

        except BinanceAPIException as e:
            print(f"Binance API error: {e}")
            return None

    def get_klines_data(self,symbol = ''):
        klines_data = self.client.futures_klines(
            symbol=symbol,
            interval=self.interval,
            startTime=self.start_time,
            endTime=self.end_time
        )
        df = pd.DataFrame(klines_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                                'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
                                                'taker_buy_quote_asset_volume', 'ignore'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    def prepare_data(self):
        df = self.get_klines_data(self.symbol)
        df = df[['close']]

        # 将数据归一化到0-1之间
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df.values)

        # 划分训练集和测试集
        training_size = int(len(scaled_data) * self.split_ratio)
        test_size = len(scaled_data) - training_size
        train_data, test_data = scaled_data[0:training_size, :], scaled_data[training_size:len(scaled_data), :]

        # 构建训练数据
        X_train, y_train = [], []
        for i in range(120, len(train_data)):
            X_train.append(train_data[i - 120:i, 0])
            y_train.append(train_data[i, 0])

        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        # 构建测试数据
        X_test, y_test = [], []
        for i in range(120, len(test_data)):
            X_test.append(test_data[i - 120:i, 0])
            y_test.append(test_data[i, 0])
        X_test, y_test = np.array(X_test), np.array(y_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        return X_train, y_train, X_test, y_test, scaler,scaled_data
    def build_model(self,mode  = 2):
        if mode ==1 :
            X_train, y_train, X_test, y_test, scaler,scaled_data = self.prepare_data()

            # 构建LSTM模型
            model = Sequential()
            model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
            model.add(LSTM(units=50))
            model.add(Dense(units=1))

            # 编译模型
            model.compile(optimizer='adam', loss='mean_absolute_error')

            return model
        if mode ==2 :
            X_train, y_train, X_test, y_test, scaler,scaled_data = self.prepare_data()

            # 构建RNN模型
            model = Sequential()
            model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], 1)))
            model.add(Dropout(0.4))
            model.add(LSTM(units=100, return_sequences=True))
            model.add(Dropout(0.4))
            model.add(LSTM(units=100))
            model.add(Dropout(0.4))
            model.add(Dense(units=1))

            # 编译模型
            model.compile(optimizer='adam', loss=rmse)

            return model
    def predict_future(self,num_days,model):

        df = self.get_klines_data()
        df = df[['close']]
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df.values)

        predictions = []
        for i in range(num_days):
            prediction_normalized = model.predict(scaled_data)

            prediction = scaler.inverse_transform(prediction_normalized)

            # 添加预测结果到结果列表
            predictions.append(prediction)

            # 更新最近数据：去掉最早的一天数据，添加预测的一天数据
            scaled_data = np.roll(scaled_data, shift=-1, axis=1)
            scaled_data[:, -1, :] = prediction_normalized

        return predictions
    def find_file(self,name, folder_path='./models//'):
        for filename in os.listdir(folder_path):
            if name in filename:
                return folder_path+filename
        return ""
    def sliding_window_prediction(self,model, X_test, days_to_predict):
        predictions = []
        recent_data = X_test[-1]  # 从X_test中获取最近的一组数据

        for _ in range(days_to_predict):
            y_pred = model.predict(np.expand_dims(recent_data, axis=0))  # 对最近的数据进行预测
            predictions.append(y_pred[0])  # 将预测结果添加到结果列表

            # 更新最近数据：去掉最早的一天数据，添加预测的一天数据
            recent_data = np.roll(recent_data, shift=-1, axis=0)
            recent_data[-1] = y_pred

        return np.array(predictions)
    def train_model(self):
        # 获取数据集
        X_train, y_train, X_test, y_test, scaler, scaled_data = self.prepare_data()
        # 训练模型
        model = self.build_model()
        model.fit(X_train, y_train, epochs=self.num_epochs, batch_size=32)

        # 预测

        import numpy as np

        # 预测未来20天的数据
        days_to_predict = 7
        future_predictions = self.sliding_window_prediction(model, X_test, days_to_predict)

        print("Future predictions:\n", future_predictions)

        # 假设已经有了X_test的预测结果（y_pred）和未来20天的预测结果（future_predictions）
        y_pred = model.predict(X_test)
        future_predictions = self.sliding_window_prediction(model, X_test, days_to_predict)

        # 将一维数组转换为二维数组
        y_test_2D = y_test.reshape(-1, 1)
        y_pred_2D = y_pred.reshape(-1, 1)
        future_predictions_2D = future_predictions.reshape(-1, 1)

        # 将预测结果反归一化
        y_test_inverse = scaler.inverse_transform(y_test_2D)
        y_pred_inverse = scaler.inverse_transform(y_pred_2D)
        future_predictions_inverse = scaler.inverse_transform(future_predictions_2D)

        # 将二维数组转换回一维数组
        y_test_inverse = y_test_inverse.squeeze()
        y_pred_inverse = y_pred_inverse.squeeze()
        future_predictions_inverse = future_predictions_inverse.squeeze()
        print(len(future_predictions_inverse))
        # 计算未来预测数据的x轴坐标
        future_x_axis = np.arange(len(y_test_inverse), len(y_test_inverse) + len(future_predictions_inverse))
        # 绘制测试数据和预测数据
        plt.figure(figsize=(12, 6))
        plt.plot(y_test_inverse, color='blue', label='Actual Data')
        plt.plot(y_pred_inverse, color='green', label='Predicted Data')
        plt.plot(future_x_axis, future_predictions_inverse, color='red', label=self.symbol + 'Future Predictions')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.show()

        model.save(modelspath + self.symbol + '_premodel.h5')

        if self.is_trend_up(future_predictions_inverse):
            print(self.symbol + "升")
        else:
            print(self.symbol + "降")

    def run_predict(self,name,days_to_predict=7):
        m = self.find_file(name=name)
        print(m)
        model = load_model(m)
        df = self.get_klines_data(name)
        df = df[['close']]
        # 将数据归一化到0-1之间
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df.values)

        X_test = scaled_data[:-1]  # 输入数据是所有数据除了最后一个数据点
        y_test = scaled_data[1:]

        y_pred = model.predict(X_test)
        future_predictions = self.sliding_window_prediction(model, X_test, days_to_predict)

        # 将一维数组转换为二维数组
        y_test_2D = y_test.reshape(-1, 1)
        y_pred_2D = y_pred.reshape(-1, 1)
        future_predictions_2D = future_predictions.reshape(-1, 1)

        # 将预测结果反归一化
        y_test_inverse = scaler.inverse_transform(y_test_2D)
        y_pred_inverse = scaler.inverse_transform(y_pred_2D)
        future_predictions_inverse = scaler.inverse_transform(future_predictions_2D)

        # 将二维数组转换回一维数组
        y_test_inverse = y_test_inverse.squeeze()
        y_pred_inverse = y_pred_inverse.squeeze()
        future_predictions_inverse = future_predictions_inverse.squeeze()
        print(len(future_predictions_inverse))
        # 计算未来预测数据的x轴坐标
        future_x_axis = np.arange(len(y_test_inverse), len(y_test_inverse) + len(future_predictions_inverse))
        # 绘制测试数据和预测数据
        return  y_test_inverse,y_pred_inverse,future_x_axis, future_predictions_inverse
        # draw.figure(figsize=(12, 6))
        # draw.plot(y_test_inverse, color='blue', label='Actual Data')
        # draw.plot(y_pred_inverse, color='green', label='Predicted Data')
        # draw.plot(future_x_axis, future_predictions_inverse, color='red', label=name + 'Future Predictions')
        # draw.xlabel('Time')
        # draw.ylabel('Value')
        # draw.legend()
        # return plt
        # plt.show()
    def is_trend_up(self,data):
        """
        判断一组数据的整体趋势是否上升

        参数：
        data: numpy数组，一组数据

        返回值：
        bool值，如果数据整体上升，则返回True；否则返回False。
        """
        return data[-1] - data[0] > 0
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

