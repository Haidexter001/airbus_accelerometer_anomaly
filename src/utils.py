import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


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

