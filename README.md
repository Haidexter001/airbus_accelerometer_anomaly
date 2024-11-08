# Airbus Accelerometer Anomaly Detection
Autoencoder training pipeline for anomalous acceleration detection in Airbus Helicopter Accelerometer dataset


## Dataset
### Description
According to the source: 

The training dataset is composed of 1677 one-minute-sequences @1024Hz of accelerometer data measured on test helicopters at various locations, in various angles (X, Y, Z), on different flights. All data has been multiplied by a factor so that absolute values are meaningless, but no other normalization procedure was carried out. All sequences are considered as normal and should be used to learn normal behaviour of accelerometer data.

The validation dataset is composed of 594 one-minute-sequences of accelerometer data measured on test helicopters at various locations, in various angles (X, Y, Z). Locations and angles may or may not be identical to those of the training dataset. Sequences are to be tested with the normal behaviour learnt from the training data to detect abnormal behaviour. The amount of abnormal sequences in the validation dataset is a priori unknown.
### Data Preparation
Each data sequence contains t * freq (60*1024 = 61440) datapoints. It is unknown whether each 1-sec 1024Hz chunk itself is a normal window of acceleration, or if the entire sequence should be considered normal, despite variances within each 1-sec chunk. 

Since this is essentially a supervised problem with known labels, we can simplify the solution by considering the entire 61440 data points as a single time series data sample, and then label each sequence as either normal (1) or abnromal (0). 

Training set -> only normal dataset
Eval set -> shuffled chunk of combined abnormal+normal dataset


## Model
### Overview
For this task, I have used an autoencoder architecture. When trained on the normal dataset, the model will learn to reconstruct normal accelerometer sensor data, albeit with some reconstruction errors. Since we're only interested in the similarity of the reconstructed signal from the decoder, we can set a threshold for this error. This way, when an abnornmal signal is passed into the autoencoder, we can expect to get an extremely high reconstruction error. This high reconstruction error is a tell-tale sign of an anomaly. 
### Architecture
I have created a simple unsupervised autoencoder transformer model with an encoder and decoder, each with 4 Dense layers. The input and ouput shapes are the length of each time-series sequence (60*1024 datapoints) The encoder will extract features of its input and generate its lower-dimensional. The decoder then takes the lower-dimensional features and tries to regenerate the original input signal as close as possible. 
### Training
For the sake of following instructions, I have created functions that combine and shuffle the abnormal and normal dataframes before preparing the train and test sets. However, we do not need the abnormal dataset for the training phase; we will only use the normal dataset to train the model, and then use the reconstruction loss to determine the quality of our input signal on the test set.

The training dataset is exclusively the normal sensor data. The validation data is a portion of a combined and randomly shuffled normal+abnormal dataset. 
### Evaluation
This binary classification problem is evaluated using precision-recall and accuracy metrics from scikit-learn.

The criteria set for an abnormal signal is a reconstruction error of two standard deviations from the MAE of the training loss. The loss of the test set (with both anomalous and shuffled data) is then compared to the reconstruction error to evaluate the accuracy of the model.

## Scripts
main.py -> main script to prepare data, train, and evaluate the model
config.yaml -> configuration markup file for initializing model training and declaring training callbacks
data_prep.py -> helper functions for preparing the HDF5 datasets
model.py -> contains Keras-based autoencoder model architecture class
evaluation.py -> evaluation functions to generate evaluation metrics for trained model
trainer.py -> functions for compiling and training the model
utils.py -> helper functions for data visualization

## Training
The training begins from setting the necessary parameters in the config file. After that, all you need to do is run main.py and the model training should start. Once the model has trained, it will save the model and evaluate its accuracy, precision, and recall. 
execute main.py to train 
