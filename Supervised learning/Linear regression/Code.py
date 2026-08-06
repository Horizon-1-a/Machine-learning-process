#%%
import sys
from sklearn.cluster import 
import numpy as np
import matplotlib.pyplot as plt

sys.stdin = open('input.txt', 'r')
#sys.stdout = open('output.txt', 'w')

def run(x, y, learning_rate, epochs):
#test:
    a, n = x.shape
    x_bar = np.vstack((x, np.ones((1, n))))
    w = np.zeros((1, a + 1))
    history = [w]
    for i in range (epochs):
        error = y - w @ x_bar
        w -= -2 / n * (error @ x_bar.T) * learning_rate
        history.append(w);
    return w
np.random.seed(42)  # để kết quả tái lập được mỗi lần chạy

a, n = 3, 200                          # 3 đặc trưng (ẩn), 200 mẫu
X = np.random.randn(a, n) * 2          # (a, n)

w_true = np.array([2.0, -1.5, 0.5])    # trọng số thật
b_true = 4.0                           # bias thật

noise = np.random.randn(n) * 0.1
y = (w_true @ X + b_true + noise).reshape(1, n)   # (1, n)

W_learned = run(X, y, learning_rate=0.001, epochs=10000)

print("w_thật + b_thật:", np.append(w_true, b_true))
print("W học được:      ", W_learned.flatten())
