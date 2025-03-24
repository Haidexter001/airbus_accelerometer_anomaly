import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.models import Model

def calculate_threshold(model: Model, normal_train_data: np.ndarray, factor: int = 2) -> float:
    """
    Calculate the anomaly detection threshold based on training data.

    Args:
        model (Model): Autoencoder model.
        normal_train_data: Training data for threshold calculation.
        factor (int): Multiplier for the standard deviation. Default is 2.

    Returns:
        float: Calculated threshold.
    """
    reconstructions = model.predict(normal_train_data)
    mse = tf.keras.losses.MeanSquaredError()
    train_loss = mse(reconstructions, normal_train_data)
    mean_error = np.mean(train_loss)
    std_error = np.std(train_loss)
    threshold = mean_error + factor * std_error
    return float(threshold)

def plot_n_timeseries(df: pd.DataFrame, n: int =10, fig_size: tuple=(15,4), window: int=1024, save: bool=False):
    '''
    Plot the first n time series in pandas dataset

    Args:

    df -> pandas dataframe
    n -> number of samples to plot
    fig_size -> size of plot
    window -> sampling window for plot    
    '''
    plt.figure(figsize=fig_size)
    for i in range(0, n):
        plt.plot(df.iloc[i][:window])
    if save:
        plt.savefig("Timeseries_sample.png")

