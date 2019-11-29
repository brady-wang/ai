import keras
from keras.layers import Dense
from keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt

x_data = np.random.rand(100)
noise = np.random.normal(0,0.01,x_data.shape)
y_data = x_data * 0.1 + 0.2 + noise
plt.scatter(x_data,y_data)
#plt.show()

model = Sequential()
model.add(Dense(units=1,input_dim=1))
model.compile(optimizer='sgd',loss='mse')

X_train, Y_train = x_data[:60], x_data[:60]  # 前160组数据为训练数据集
X_test, Y_test = x_data[60:], x_data[60:]  # 后40组数据为测试数据集

for step in range(3001):
    cost = model.train_on_batch(X_train,Y_train)
    if step%500 == 0:
        print('cost',cost)

w,b = model.layers[0].get_weights()
print('w:',w,"b:",b)


# 将训练结果绘出
Y_pred = model.predict(X_test)
plt.scatter(X_test, Y_test)
plt.plot(X_test, Y_pred)
plt.show()