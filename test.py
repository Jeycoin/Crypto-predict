import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QLineEdit, QLabel, QHBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from LSTM import  CryptoPricePrediction
from PyQt5.QtWidgets import QSizePolicy,QMainWindow, QPushButton, QLabel, QTextEdit,QLineEdit, QVBoxLayout, QHBoxLayout, QWidget, QSplitter
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
# 实例化预测模型
api_key = 'gfvVH43y51jiEuzdFfNDwthxsmm06xLHCR24cGe1UVDsPVGXEfmFUeMLNOsqNFFj'
api_secret = 'h2dS20ZdM4sOBiTyxl8xSPrqdTVBHmUEOiCt95ZnICKTZJNvBEFfTOqnj0EJDrkL'
symbol = 'DOGEUSDT'
predictor = CryptoPricePrediction(api_key=api_key,api_secret=api_secret,symbol=symbol, interval='1h', start_time='2022-09-01', end_time='2023-05-5', split_ratio=0.5, num_epochs=300)



class MplCanvas(FigureCanvas):
    def __init__(self):
        fig = Figure()
        self.ax = fig.add_subplot(111)
        super().__init__(fig)


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Stock Prediction")

        self.stock_analysis_button = QPushButton("股票分析")
        self.buy_button = QPushButton("买入")
        self.sell_button = QPushButton("卖出")
        self.stock_prediction_button = QPushButton("股票预测")

        self.stock_analysis_button.clicked.connect(self.display_line_chart)

        self.currency_pair_label = QLabel("货币对:")
        self.currency_pair_edit = QLineEdit()

        self.canvas = MplCanvas()

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.buy_button)
        button_layout.addWidget(self.sell_button)

        input_layout = QHBoxLayout()
        input_layout.addWidget(self.currency_pair_label)
        input_layout.addWidget(self.currency_pair_edit)

        left_layout = QVBoxLayout()
        left_layout.addLayout(input_layout)
        left_layout.addLayout(button_layout)
        left_layout.addWidget(self.stock_prediction_button)
        left_layout.addWidget(self.stock_analysis_button)

        left_container = QWidget()
        left_container.setLayout(left_layout)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_container)

        # Add a widget for the text box area
        right_container = QWidget()
        right_layout = QVBoxLayout()

        right_layout.addWidget(self.canvas)
        right_layout.addStretch(1)


        self.text_box = QTextEdit()
        self.text_box.append('message:')
        self.text_box.setReadOnly(True)

        font = QFont("宋体", 16)
        self.text_box.setFont(font)
        right_layout.addWidget(self.text_box)
        right_container.setLayout(right_layout)
        splitter.addWidget(right_container)

        splitter.setSizes([100, 900])

        self.setCentralWidget(splitter)

    def display_line_chart(self):
        name=self.currency_pair_edit.text()

        y_test_inverse,y_pred_inverse,future_x_axis, future_predictions_inverse = predictor.run_predict(name,days_to_predict=7)
        # 清除之前的图形
        self.canvas.ax.clear()
        self.text_box.clear()
        self.canvas.ax.plot(y_test_inverse, color='blue', label='Actual Data')
        self.canvas.ax.plot(y_pred_inverse, color='green', label='Predicted Data')
        self.canvas.ax.plot(future_x_axis, future_predictions_inverse, color='red', label=name + 'Future Predictions')

        if predictor.is_trend_up(future_predictions_inverse):
            self.text_box.append(name+"升")
        else:
            self.text_box.append(name + "降")


        self.canvas.ax.set_xlabel('Time')
        self.canvas.ax.set_ylabel('Value')

        self.canvas.ax.legend()
        self.canvas.draw()





app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec_())
