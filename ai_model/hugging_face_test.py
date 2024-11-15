from huggingface_hub import hf_hub_download
import onnxruntime
import numpy as np
from tensorflow.keras.preprocessing import image

model_path = hf_hub_download(repo_id="HealthMonitoringSystem/Breast_Cancer_Prediction_AI_Model", filename="model.onnx")
session = onnxruntime.InferenceSession(model_path)


def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # normalizacja
    return img_array

image_path = "./dataset/malignant/malignant (1).png"
img_array = preprocess_image(image_path)

inputs = {session.get_inputs()[0].name: img_array}
predictions = session.run(None, inputs)

class_names = ['benign', 'malignant', 'normal']
predicted_class = class_names[np.argmax(predictions)]
confidence = np.max(predictions)

print(f"Predicted class: {predicted_class}, confidence: {confidence * 100:.2f}%")