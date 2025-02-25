import tensorflow as tf
import numpy as np

# Generate sample data
X = np.array([1, 2, 3, 4, 5], dtype=np.float32)
y = np.array([2, 4, 6, 8, 10], dtype=np.float32)  # y = 2x

# Define a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

# Compile model
model.compile(optimizer='sgd', loss='mean_squared_error')

# Train model
model.fit(X, y, epochs=500, verbose=0)

# Predict a new value
print("Prediction for x=6:", model.predict([6])[0][0])
