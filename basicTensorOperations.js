import tensorflow as tf

# Create tensors
a = tf.constant(5)
b = tf.constant(3)

# Perform operations
add = tf.add(a, b)
mul = tf.multiply(a, b)

print("Addition:", add.numpy())
print("Multiplication:", mul.numpy())
