import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np


MODEL_PATH = "./"
BREAST_USG_MODEL_H5 = "breast_ultrasound_model.h5"

# Load model (HDF5 or SavedModel)
model = tf.keras.models.load_model(BREAST_USG_MODEL_H5)  # HDF5


print(f"Model successfully loaded! Model Summary:")
model.summary()

# Clasification
class_names = ['benign', 'malignant', 'normal']  # classes / categories

# test dataset
test_image_dir = "./dataset/test"


def predict_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # normalisation


    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)

    return predicted_class, confidence

test_image_dir = [ ("./dataset/benign/", "benign (25).png"), ("./dataset/test/malignant/", "malignant (71).png"), ("./dataset/test/normal/","normal (71).png")]

for image_path, image_filename in test_image_dir:
    PATH = image_path + image_filename
    predicted_class, confidence = predict_image(PATH)
    print(f"Image: {image_filename} -> Predicted: {predicted_class} with confidence: {confidence * 100:.2f}%")


