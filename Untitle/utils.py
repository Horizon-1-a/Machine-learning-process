import numpy as np

# x: feature x sample
def svd(x):
    sigma, v = np.linalg.eigh(x.T @ x)
    idx = np.argsort(sigma)[::-1]
    v = v[:, ::-1]
    return v
def reduce_data(x, k):
    v = svd(x.T)
    pc_k = v[:, :k]
    x_reduced = pc_k.T @ x
    return x_reduced
def one_hot_encode(y, k):
    if y.ndim == 1:
        y = np.eye(k)[y].T
    return y

def relu(z):
    return np.maximum(z, 0)
def d_relu(z):
    return (z > 0).astype(float)
def softmax(z):
    z = z - np.max(z, axis= 0, keepdims = True)
    z = np.exp(z)
    z = z / np.sum(z, axis = 0, keepdims = True)
    return z;
def train(x, y, epochs, learning_rate, lim, rate):
    a, n = x.shape
    lim = min(lim, n)
    n = min(lim, n)
    print(f'lim = {lim}')

    #initializing
    idx = np.random.permutation(lim)
    x = x[:, idx]
    y = y[idx]
    y = one_hot_encode(y, 10)

    n_train = int(n * rate)
    n_validate = n - n_train

    x_train = x[:, :n_train]
    x_bar = np.vstack((x_train, np.ones((1, n_train))))

    x_validate = x[:, n_train:]
    x_validate = np.vstack((x_validate, np.ones((1, n_validate))))
    y_train = y[:, :n_train]
    y_validate = y[:, n_train: ]

    w1 = np.random.randn(256, a + 1) * np.sqrt(2 / a)
    w2 = np.random.randn(10, 257) * np.sqrt(2 / 256)
    #Saving best option 
    w1_best = w1.copy()
    w2_best = w2.copy()
    acc_best_validation = 0.0
    acc_training = 0.0
    acc_validation = 0.0

    #main_func
    for i in range (epochs):
        shuffle = np.random.permutation(n_train)

        for j in range (0, n_train, 50):
            end = min(j + 50, n_train)
            n_batch = end - j
            idx = shuffle[j: end]
            x_cur = x_bar[:, idx]
            y_cur = y_train [:, idx]

            #forward_pass
            z1 = w1 @ x_cur
            a1 = relu(z1)
            a1_bar = np.vstack((a1, np.ones((1, n_batch))))
            z2 = w2 @ a1_bar
            a2 = softmax(z2)

            #back_prop
            dz2 = 1 / n_batch * (a2 - y_cur)
            dw2 = dz2 @ a1_bar.T
            da1 = w2[:, :-1].T @ dz2
            dz1 = d_relu(z1) * da1
            dw1 = dz1 @ x_cur.T

            #update
            w1 -= dw1 * learning_rate
            w2 -= dw2 * learning_rate
        #training:
        z1_t = w1 @ x_bar
        a1_t = relu(z1_t)
        a1_bar_t = np.vstack((a1_t, np.ones((1, n_train))))
        z2_t = w2 @ a1_bar_t
        a2_t = softmax(z2_t)
        y_pred_t = np.argmax(a2_t, axis= 0)
        y_truth_t = np.argmax(y_train, axis = 0)
        acc_training = np.mean(y_pred_t == y_truth_t)

        #validation
        z1_v = w1 @ x_validate
        a1_v = relu(z1_v)
        a1_bar_v = np.vstack((a1_v, np.ones((1, n_validate))))
        z2_v = w2 @ a1_bar_v
        a2_v = softmax(z2_v)
        y_pred_v = np.argmax(a2_v, axis= 0)
        y_truth_v = np.argmax(y_validate, axis= 0)
        acc_validation = np.mean(y_pred_v == y_truth_v)

        print(f'Epoch {i}th: acc_training = {acc_training}, acc_validation = {acc_validation}')

        if (acc_validation > acc_best_validation):
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

            

