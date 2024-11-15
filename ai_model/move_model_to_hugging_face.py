import tf2onnx
from tensorflow.keras.models import load_model
from huggingface_hub import HfApi, Repository

from constants import  BREAST_USG_MODEL_H5, ONNX_MODEL_PATH

# Load the model from h5 format
model = load_model(BREAST_USG_MODEL_H5)

onnx_model = tf2onnx.convert.from_keras(model)

onnx_model.save(ONNX_MODEL_PATH)

model_path = ONNX_MODEL_PATH
repo_name = "Breast_Cancer_Prediction_AI_Model"

api = HfApi()
repo_url = api.create_repo(repo_id=repo_name, exist_ok=True)

repo = Repository(local_dir=repo_name, clone_from=repo_url)
repo.lfs_track(["*.onnx"])
repo.push_to_hub(commit_message="Upload ONNX model")