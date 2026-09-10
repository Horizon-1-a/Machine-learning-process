import numpy as np
from utils import reduce_data, train, predict, svd
import tensorflow as tf


(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
# 2. Normalize
X_train = X_train.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0

# 3. Flatten + chuyển sang features × samplex 
X_train = X_train.reshape(X_train.shape[0], -1).T
X_test = X_test.reshape(X_test.shape[0], -1).T
X_train = X_train[:, :30000]
y_train = y_train[:30000]
mean = np.mean(X_train, axis= 1, keepdims= True)
X_train = X_train - mean
X_test = X_test - mean
v_train = svd(X_train.T)
pc_k = v_train[:, :50]
X_train = pc_k.T @ X_train
X_test = pc_k.T @ X_test




w1, w2 = train(X_train, y_train, epochs=50, learning_rate = 0.1, lim = 10000, rate = 0.8)
predict(X_test, w1, w2, y_test)
