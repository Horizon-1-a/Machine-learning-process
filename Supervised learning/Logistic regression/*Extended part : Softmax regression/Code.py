#%%
import sys
import numpy as np
import matplotlib.pyplot as plt

#sys.stdin = open('input.txt', 'r')
sys.stdout = open('output.txt', 'w')
def one_hot_encode (y, k):
    if y.ndim == 1 :
        #Tạo mảng one_hot (dạng[0, 1, ..., 0].T)
        y = np.eye(k)[y].T 
    return y
def probalitize(x):
    e = np.exp(x)
    s = np.sum(e, axis = 0)
    return (e / s)
def run(x, y, k, epochs, learning_rate):
    a, n = x.shape
    x_bar = np.vstack((x, np.ones((1, n))))
    w = np.zeros((k, a + 1))
    y = one_hot_encode(y, k)
    history = [w.copy()]
    for i in range (epochs):
        p = probalitize(w @ x_bar)
        error = p - y
        w -= 1 / n * (error @ x_bar.T) * learning_rate
        history.append(w.copy())
    return w, history;
np.random.seed(42)

# Mẫu dữ liệu (2 đặc trưng, 9 mẫu) chia thành 3 nhóm
x_test = np.array([
    [1.0, 1.2, 1.5,  5.0, 5.2, 5.5,  9.0, 9.2, 9.5], # Feature 1
    [1.0, 0.8, 1.1,  5.0, 4.8, 5.1,  1.0, 1.2, 0.9]  # Feature 2
])
y_test = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]) # 3 lớp: 0, 1, 2

# Train model
W_fit, _ = run(x_test, y_test, k=3, epochs=2000, learning_rate=0.1)
print (W_fit)




    

    

