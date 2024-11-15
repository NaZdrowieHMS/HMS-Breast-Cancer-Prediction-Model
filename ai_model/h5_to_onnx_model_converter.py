import tf2onnx
import tensorflow as tf
from constants import BREAST_USG_MODEL_H5, ONNX_MODEL_PATH

# Load the trained model (HDF5 format)
model = tf.keras.models.load_model(BREAST_USG_MODEL_H5)

# Define input signature for the model
spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)

# Convert the model to ONNX
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)

# Save the ONNX model to file
with open(ONNX_MODEL_PATH, "wb") as f:
    f.write(onnx_model.SerializeToString())

print("Model successfully converted and saved to 'model.onnx'")