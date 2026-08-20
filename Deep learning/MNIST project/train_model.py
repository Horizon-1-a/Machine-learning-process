import numpy as np
from utils import train_model, predict
import tensorflow as tf


(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
# 2. Normalize
X_train = X_train.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0

# 3. Flatten + chuyển sang features × samples
X_train = X_train.reshape(X_train.shape[0], -1).T
X_test = X_test.reshape(X_test.shape[0], -1).T

w1, w2 = train_model(X_train, y_train, epochs=50, learning_rate = 0.1, lim = 10000)
predict(X_test, w1, w2, y_test)


