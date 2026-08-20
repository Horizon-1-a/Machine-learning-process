import numpy as np
import matplotlib.pyplot as plt
# Ví dụ với bài toán XOR dùng softmax để tổng quát
def Relu(x):
    return np.maximum(0, x)
def deri_Relu(z):
    return (z > 0).astype(float)
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)
def one_hot_encode(y, k):
    if y.ndim == 1:
        y = np.eye(k)[y].T
    return y
def run (x, y, epochs, learning_rate):
    #initialize:
    a, n = x.shape
    y = one_hot_encode(y, 2)
    w1 = np.random.randn(32, a + 1) * 0.1
    w2 = np.random.randn(2, 33) * 0.1
    x_bar = np.vstack((x, np.ones((1, n))))

    #main function:
    for i in range (epochs):
        #forward_pass:
        z1 = w1 @ x_bar
        a1 = Relu(z1)
        a1_bar = np.vstack((a1, np.ones((1, n))))
        z2 = w2 @ a1_bar
        a2 = softmax(z2)

        #back_prop
        dz2 = 1 / n * (a2 - y)
        dw2 = dz2 @ a1_bar.T
        da1 = w2[:, :-1].T @ dz2
        dz1 = da1 * deri_Relu(z1)
        dw1 = dz1 @ x_bar.T

        #update:
        w1 -= dw1 * learning_rate
        w2 -= dw2 * learning_rate
    return w1, w2
def predict(test, w1, w2):
    p, n = test.shape
    test_bar = np.vstack((test, np.ones((1, n))))
    z1 = w1 @ test_bar
    a1 = Relu(z1)
    a1_bar = np.vstack((a1, np.ones((1, n))))
    z2 = w2 @ a1_bar
    a2 = softmax(z2)
    return a2
x = np.array([[1, 1, 0, 0],
             [1, 0, 1, 0]])
y = np.array([[0, 1, 1, 0], 
              [1, 0, 0, 1]])
w1, w2 = run(x, y, 10000, 0.1)
test = np.random.rand(2, 1)
res = predict(test, w1, w2)
print (res);


