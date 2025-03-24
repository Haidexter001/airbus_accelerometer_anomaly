# Airbus Accelerometer Anomaly Detection
LSTM training package for my attempt at the anomalous acceleration detection in Airbus Helicopter Accelerometer dataset

## Dependencies
Tested with:

Python=3.11 
Tensorflow=2.18
numpy=2.2.4
pandas=2.2.3

## Dataset
### Description
**From the source:**
https://www.research-collection.ethz.ch/handle/20.500.11850/415151

## Model
For this particular challenge, I've used a custom-built LSTM model. 
This is an unsupervised learning problem. The model is trained only on the "normal" acceleration data. 

## Training & Evaluation
- Download the train and validation (normal and abnormal detections) to data/
- Set up the variables in config.yaml
- Run model.py

To train on your own data, it must be in Hd5 pandas format.

Additional features (visualization, model freezing, ONNX deployment) coming soon...
