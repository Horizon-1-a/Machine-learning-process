## Optimizing the error rate of validation dataset

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
    idx = np.random.permutation(lim)

    x = x[:, idx]
    y = y[idx]
    a,_ = x.shape
    validation_size = int(lim * 0.15)
    training_size = lim - validation_size
    x_validation = x[:, :validation_size]
    x_validation = np.vstack((x_validation, np.ones((1, validation_size))))
    x_training = x[:, validation_size : lim]
    x_bar = np.vstack((x_training, np.ones((1, training_size))))


    y_validation_ex = y[:validation_size]
    y_training_ex = y[validation_size:lim]
    y = y[validation_size:lim]
    y = one_hot_encode(y, 10)

    w1 = np.random.randn(256, a + 1) * np.sqrt(2 / a)
    w2 = np.random.randn(10, 257) * np.sqrt(2 / 256)
    #Saving best option
    w1_best = w1.copy()
    w2_best = w2.copy()
    acc_best_training = 0.0
    acc_best_validation = 0.0
    acc_training = 0.0
    acc_validation = 0.0
    #main function
    for i in range(epochs):
        shuffle = np.random.permutation(training_size)
        for j in range (0, training_size, 50):
            batch_size = min(training_size, j + 50) - j
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
        #update
        if True:
            #training:
            z1_t = w1 @ x_bar
            a1_t = relu(z1_t)
            a1_bar_t = np.vstack((a1_t, np.ones((1, training_size))))
            z2_t = w2 @ a1_bar_t
            a2_t = softmax(z2_t)
            y_pred_t = np.argmax(a2_t, axis= 0)
            acc_training = np.mean(y_pred_t == y_training_ex)
            #validation
            z1_v = w1 @ x_validation
            a1_v = relu(z1_v)
            a1_bar_v = np.vstack((a1_v, np.ones((1, validation_size))))
            z2_v = w2 @ a1_bar_v
            a2_v = softmax(z2_v)
            y_pred_v = np.argmax(a2_v, axis= 0)
            acc_validation = np.mean(y_pred_v == y_validation_ex)

            print(f'Epoch {i}: acc_training = {acc_training}, acc_validation = {acc_validation}')

            if (acc_validation >= acc_best_validation):
                acc_best_training = acc_training
                acc_best_validation = acc_validation
                w1_best = w1.copy()
                w2_best = w2.copy()
                print (f'--> Updating best so far, acc_training = {acc_training}, acc_validation = {acc_validation}')
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
