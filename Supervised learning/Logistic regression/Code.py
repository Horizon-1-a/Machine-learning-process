#%%
import sys
import numpy as np
import matplotlib.pyplot as plt

#sys.stdin = open('input.txt', 'r')
#sys.stdout = open('output.txt', 'w')
def sigmoid (w, x):
    z = w @ x
    return 1 / (1 + np.exp(-z))
def run(x, y, learning_rate, epochs):
#test:
    if x.ndim == 1:
       x = x.reshape(1, -1)
    if y.ndim == 1:
       y =  y.reshape(1, -1)
    a, n = x.shape
    x_bar = np.vstack((x, np.ones((1, n))))
    w = np.zeros((1, a + 1))
    history = [w]
    for i in range (epochs):
        error = sigmoid (w, x_bar) - y
        w -= 1 / n * (error @ x_bar.T) * learning_rate
        history.append(w.copy())
    return w, history;

x = np.array([1.0, 2.0, 3.0, 4.0, 8.0, 9.0, 10.0, 11.0])
y = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0,  1.0,  1.0])
w , history = run(x, y, 0.001, 10000)
# --- Visualization (using imported matplotlib) ---
x_test = np.linspace(0, 12, 100).reshape(1, -1)
x_test_bar = np.vstack((x_test, np.ones((1, x_test.shape[1]))))
y_prob = sigmoid(w, x_test_bar)

plt.scatter(x, y, color='red', label='Data points')
plt.plot(x_test[0], y_prob[0], color='blue', label='Logistic Fit')
plt.axhline(0.5, color='gray', linestyle='--', label='Decision Boundary (0.5)')
plt.xlabel('x')
plt.ylabel('Probability')
plt.title('Logistic Regression Fit')
plt.legend()
plt.grid(True)
plt.show()
