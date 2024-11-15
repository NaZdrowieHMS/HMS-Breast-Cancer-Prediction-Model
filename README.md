# HMS-Breast-Cancer-Prediction-Model

AI model used by Health Monitoring System to predict breast cancer based on USG

## Model Description

    Model avaiable and deployed on Hugging Face:
1. https://huggingface.co/HealthMonitoringSystem/Breast_Cancer_Prediction_AI_Model/tree/main
2. https://api-inference.huggingface.co/models/HealthMonitoringSystem/Breast_Cancer_Prediction_AI_Model

## Requirements

To prepare and train breast cancer prediction AI model you need

1. [Python 3.12 or higher](https://www.python.org/downloads/release/)

2. Install the required dependencies by running the following command:
```
pip install -r requirements.txt
```


## Train the model

1. Prepare the training, validation, and test sets


2. Run the `model.py` file


## Convert the model format from h5 to onnx
The SavedModel format is used by HealthMonitoringSystemApplication java backend

1. Run the `h5_to_onnx_model_converter.py` file
