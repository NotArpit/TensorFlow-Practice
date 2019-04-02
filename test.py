import tensorflow as tf
import numpy as np
from tensorflow import keras

# Create the Neural Network:
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
# Compile the Neural Network using Stochastic Gradient Descent and a 
# Mean Squared Error Cost Function:
model.compile(optimizer='sgd', loss='mean_squared_error')
# Create arbitrary training set:
xs = np.array([1,2,3,4,5,6], dtype=int)
# Create arbitrary output values (continuous valued output):
ys = np.array([2,4,6,8,10,12], dtype=int)
# Fit the model (i.e. Minimise mean_squared_error using sgd):
model.fit(xs, ys, epochs=500)
# Predict model for arbitrary previously unseen input:
print(model.predict([10.0]))