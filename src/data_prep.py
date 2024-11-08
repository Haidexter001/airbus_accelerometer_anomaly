import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

def extract_raw_data(df_normal_path: str, df_abnormal_path: str, add_labels: bool=True, shuffle: bool=True) -> np.ndarray:
    df_normal = pd.read_hdf(df_normal_path, header=None, sep=" ")
    df_abnormal = pd.read_hdf(df_abnormal_path, header=None, sep=" ")
    if add_labels:
        df_normal['y']=1
        df_abnormal['y']=0
    combined_df = pd.concat([df_normal, df_abnormal], ignore_index=True)
    if shuffle:
        shuffled_df = combined_df.sample(frac=1).reset_index(drop=True)
        raw_data = shuffled_df.values
        return raw_data
    else:
        raw_data = combined_df.values
        return raw_data
    
def generate_train_test_split(raw_data: np.ndarray, test_size: float = 0.2, random_state: int = 21, labeled: bool=True) -> tuple:
    if labeled:
        labels = raw_data[:, -1]
        data = raw_data[:, 0:-1]

    train_split, test_split, train_labels, test_labels = train_test_split(
    data, labels, test_size=test_size, random_state=random_state)
    
    def _normalize_train_test_split(train_data: np.ndarray, test_data: np.ndarray) -> tuple:
            min_val = tf.reduce_min(train_data)
            max_val = tf.reduce_max(train_data)

            train_data = (train_data - min_val) / (max_val - min_val)
            test_data = (test_data - min_val) / (max_val - min_val)

            train_data = tf.cast(train_data, tf.float32)
            test_data = tf.cast(test_data, tf.float32)
    
            return train_data, test_data
    
    train_data, test_data = _normalize_train_test_split(train_split, test_split)
    train_labels = train_labels.astype(bool)
    test_labels = test_labels.astype(bool)

    normal_train_data = train_data[train_labels]

    anomalous_train_data = train_data[~train_labels]
    anomalous_test_data = test_data[~test_labels]

    return normal_train_data, anomalous_train_data, test_data, anomalous_test_data, test_labels