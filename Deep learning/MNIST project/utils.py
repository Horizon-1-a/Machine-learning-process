import numpy as np

def relu(z):
    return np.maximum(z, 0)
def d_relu(z):
    return (z > 0).astype(float)
def softmax(x):
    x = np.exp(x - np.max(x, axis= 0, keepdims= True))
    s = np.sum(x, axis = 0, keepdims= True)
    return (x / s)
def one_hot_encode(y, k):
    if y.ndim == 1:
        y = np.eye(k)[y].T
    return y
def train_model(x, y, epochs, learning_rate, lim):
    print (f'Training sample = {lim}')
    #initializing
    x = x[:, :lim]
    a, n = x.shape
    x_bar = np.vstack((x, np.ones((1, n))))
    y1 = y[:lim]
    y = one_hot_encode(y, 10)
    y = y[:, :lim]
    w1 = np.random.randn(256, a + 1) * np.sqrt(2 / a)
    w2 = np.random.randn(10, 257) * np.sqrt(2 / 256)
    #Saving best option
    w1_best = w1.copy()
    w2_best = w2.copy()
    acc_best = 0.0
    #main function
    for i in range(epochs):
        shuffle = np.random.permutation(n)
        for j in range (0, n, 50):
            batch_size = min(n, j + 50) - j
            idx = shuffle[j:j + batch_size]
            x_cur = x_bar[:, idx]
            y_cur = y[:, idx]
            #forward pass
            z1 = w1 @ x_cur
            a1 = relu(z1)
            a1_bar = np.vstack((a1, np.ones((1, batch_size))))
            z2 = w2 @ a1_bar
            a2 = softmax(z2)

            #backpropagation
            dz2 = 1 / batch_size * (a2 - y_cur)
            dw2 = dz2 @ a1_bar.T
            da1 = w2[:, :-1].T @ dz2
            dz1 = d_relu(z1) * da1
            dw1 = dz1 @ x_cur.T

            #update
            w1 -= dw1 * learning_rate
            w2 -= dw2 * learning_rate

        if True:
            z1 = w1 @ x_bar
            a1 = relu(z1)
            a1_bar = np.vstack((a1, np.ones((1, n))))
            z2 = w2 @ a1_bar
            a2 = softmax(z2)
            loss = -np.mean(np.sum(y * np.log(a2 + 1e-8), axis=0))
            y_pred = np.argmax(a2, axis= 0)
            acc = np.mean(y_pred == y1)

            print (f'epochs i = {i}: Loss = {loss}, accuracy = {acc}')
            if acc > acc_best:
                acc_best = acc
                w1_best = w1
                w2_best = w2
                print (f'--> Updating best so far, accuracy = {acc_best}')
    return w1_best, w2_best
def predict(input, w1, w2, test):
    #forward pass
    a, n = input.shape
    input = np.vstack((input, np.ones((1, n))))
    z1 = w1 @ input
    a1 = relu(z1)
    a1 = np.vstack((a1, np.ones((1, n))))
    z2 = w2 @ a1
    a2 = softmax(z2)

    #accuracy
    y_pred = np.argmax(a2, axis = 0)
    acc = np.mean(y_pred == test)
    print ("accuracy:", acc)
