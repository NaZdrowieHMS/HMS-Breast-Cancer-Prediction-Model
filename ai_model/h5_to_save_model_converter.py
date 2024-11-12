import tensorflow as tf
from tensorflow.keras.models import load_model
from constants import JAVA_MODEL_PATH, BREAST_USG_MODEL_H5

# Load the model from h5 format
model = load_model(BREAST_USG_MODEL_H5)

# Export to SavedModel format (Used by Java api)
model.export(JAVA_MODEL_PATH)

print(f'Model saved in SavedModel format at: {JAVA_MODEL_PATH}')