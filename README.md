# gradient-descent-homework
python梯度下降

import numpy as np

x = np.array([1, 2, 3, 4, 5, 6, 7])
y = np.array([2.2, 3.8, 5.1, 6.9, 7.8, 9.2, 10.9])

w = 0.0  
b = 0.0  

learning_rate = 0.0001
epoch = 10000

print("===== 开始训练 =====")
for i in range(epoch):
    y_pred = w * x + b
    loss = np.mean((y_pred - y) ** 2)

    dw = np.mean(2 * (y_pred - y) * x)
    db = np.mean(2 * (y_pred - y))
    w = w - learning_rate * dw
    b = b - learning_rate * db

    if i % 2000 == 0:
        print(f"第{i}轮 | 损失={loss:.2f} | w={w:.4f} | b={b:.4f}")

print("\n===== 训练完成 =====")
print(f"拟合直线方程：y = {w:.4f} * x + {b:.4f}")

x_test = 8
y_test_pred = w * x_test + b
print(f"\n当 x = {x_test} 时，预测 y = {y_test_pred:.2f}")
