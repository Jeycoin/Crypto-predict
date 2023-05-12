from LSTM import  CryptoPricePrediction
import matplotlib.pyplot as plt

modelspath = './models//'

# 实例化预测模型
api_key = 'gfvVH43y51jiEuzdFfNDwthxsmm06xLHCR24cGe1UVDsPVGXEfmFUeMLNOsqNFFj'
api_secret = 'h2dS20ZdM4sOBiTyxl8xSPrqdTVBHmUEOiCt95ZnICKTZJNvBEFfTOqnj0EJDrkL'
symbol = 'TOMOUSDT'
predictor = CryptoPricePrediction(api_key=api_key,api_secret=api_secret,symbol=symbol, interval='1h', start_time='2022-10-01', end_time='2023-05-05', split_ratio=0.5, num_epochs= 499)

# print(predictor.get_crypto_prices())

predictor.train_model()
#
# # 获取数据集
# X_train, y_train, X_test, y_test, scaler,scaled_data = predictor.prepare_data()
#
# # 训练模型
# model = predictor.build_model()
# model.fit(X_train, y_train,epochs=predictor.num_epochs, batch_size=32)
#
#
# # 预测
#
#
#
# import numpy as np
#
#
#
# # 预测未来20天的数据
# days_to_predict = 7
# future_predictions = sliding_window_prediction(model, X_test, days_to_predict)
#
# print("Future predictions:\n", future_predictions)
#
#
#
# # 假设已经有了X_test的预测结果（y_pred）和未来20天的预测结果（future_predictions）
# y_pred = model.predict(X_test)
# future_predictions = sliding_window_prediction(model, X_test, days_to_predict)
#
#
# # 将一维数组转换为二维数组
# y_test_2D = y_test.reshape(-1, 1)
# y_pred_2D = y_pred.reshape(-1, 1)
# future_predictions_2D = future_predictions.reshape(-1, 1)
#
# # 将预测结果反归一化
# y_test_inverse = scaler.inverse_transform(y_test_2D)
# y_pred_inverse = scaler.inverse_transform(y_pred_2D)
# future_predictions_inverse = scaler.inverse_transform(future_predictions_2D)
#
# # 将二维数组转换回一维数组
# y_test_inverse = y_test_inverse.squeeze()
# y_pred_inverse = y_pred_inverse.squeeze()
# future_predictions_inverse = future_predictions_inverse.squeeze()
# print(len(future_predictions_inverse))
# # 计算未来预测数据的x轴坐标
# future_x_axis = np.arange(len(y_test_inverse), len(y_test_inverse) + len(future_predictions_inverse))
# # 绘制测试数据和预测数据
# plt.figure(figsize=(12, 6))
# plt.plot(y_test_inverse, color='blue', label='Actual Data')
# plt.plot(y_pred_inverse, color='green', label='Predicted Data')
# plt.plot(future_x_axis, future_predictions_inverse, color='red', label=symbol+'Future Predictions')
# plt.xlabel('Time')
# plt.ylabel('Value')
# plt.legend()
# plt.show()
#
#
# model.save(modelspath+symbol+'_premodel.h5')
#
# if is_trend_up(future_predictions_inverse):
#     print(symbol+"升")
# else:
#     print(symbol + "降")

