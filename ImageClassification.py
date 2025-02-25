import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np
from PIL import Image
import requests
from io import BytesIO

# Load MobileNetV2 model
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Load an image
url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/800px-Cat03.jpg'
response = requests.get(url)
image = Image.open(BytesIO(response.content)).resize((224, 224))

# Preprocess the image
img_array = np.array(image)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

# Predict
predictions = model.predict(img_array)
label = decode_predictions(predictions, top=1)[0][0]

print("Predicted label:", label[1], "with confidence", label[2])
