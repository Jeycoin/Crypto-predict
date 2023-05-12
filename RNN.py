import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM,Dropout
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from binance.client import Client
class CryptoPricePrediction:

    def __init__(self, api_key, api_secret,symbol: str, interval: str, start_time: str, end_time: str, split_ratio: float, num_epochs: int):
        self.symbol = symbol
        self.interval = interval
        self.start_time = start_time
        self.end_time = end_time
        self.split_ratio = split_ratio
        self.num_epochs = num_epochs
        self.client = Client(api_key, api_secret)
    def get_klines_data(self):
        klines_data = self.client.futures_klines(
            symbol=self.symbol,
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
        df = self.get_klines_data()
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
        for i in range(60, len(train_data)):
            X_train.append(train_data[i - 60:i, 0])
            y_train.append(train_data[i, 0])
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        # 构建测试数据
        X_test, y_test = [], []
        for i in range(60, len(test_data)):
            X_test.append(test_data[i - 60:i, 0])
            y_test.append(test_data[i, 0])
        X_test, y_test = np.array(X_test), np.array(y_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        return X_train, y_train, X_test, y_test, scaler

    def build_model(self):
        X_train, y_train, X_test, y_test, scaler = self.prepare_data()

        # 构建RNN模型
        model = Sequential()
        model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=100, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=100))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))

        # 编译模型
        model.compile(optimizer='adam', loss='mean_squared_error')

        return model

    def train_model(self):
        X_train, y_train, X_test, y_test, scaler = self.prepare_data()
        model = self.build_model()

        # 训练模型
        model.fit(X_train, y_train, epochs=self.num_epochs, batch_size=32)

        return model

    def predict(self):
        X_train, y_train, X_test, y_test, scaler = self.prepare_data()
        model = self.train_model()

        # 预测
        y_pred = model.predict(X_test)

        # 反归一化
        y_pred = scaler.inverse_transform(y_pred)
        y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

        # 可视化预测结果
        plt.plot(y_test, label='True')
        plt.plot(y_pred, label='Predicted')
        plt.title(f'{self.symbol} Price Prediction')
        plt.legend()
        plt.show()


api_key = 'gfvVH43y51jiEuzdFfNDwthxsmm06xLHCR24cGe1UVDsPVGXEfmFUeMLNOsqNFFj'
api_secret = 'h2dS20ZdM4sOBiTyxl8xSPrqdTVBHmUEOiCt95ZnICKTZJNvBEFfTOqnj0EJDrkL'
symbol = 'ETCUSDT'
predictor = CryptoPricePrediction(api_key=api_key,api_secret=api_secret,symbol=symbol, interval='1h', start_time='2022-09-01', end_time='2023-04-30', split_ratio=0.5, num_epochs=300)
predictor.predict()
