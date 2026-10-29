import numpy as np
from utils import reduce_data, train, predict, svd
import tensorflow as tf


(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
# 2. Normalize
X_train = X_train.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0

# 3. Flatten + chuyển sang features × sample
X_train = X_train.reshape(X_train.shape[0], -1).T
X_test = X_test.reshape(X_test.shape[0], -1).T
# center data by using the mean of train data
n_train = X_train.shape[0]

X_validate = X_train[:, :int(n_train * 0.2)]
Y_validate = y_train[:int(n_train * 0.2)]

X_train = X_train[:, int(n_train * 0.2):]
y_train = y_train[int(n_train * 0.2):] 

#center



w1, w2 = train(X_train, y_train, X_validate, Y_validate, epochs=50, learning_rate = 0.1, lim = 10000)
predict(X_test, w1, w2, y_test)
