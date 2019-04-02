import tensorflow as tf
import numpy as np
from tensorflow import keras

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')
xs = np.array([1,2,3,4,5,6], dtype=int)
ys = np.array([2,4,6,8,10,12], dtype=int)
model.fit(xs, ys, epochs=500)
print(model.predict([10.0]))