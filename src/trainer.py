import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model

def compile_model(model: Model, lr: float, loss: str):
    '''
    Compiles model with Adam optimizer and custom loss function for training

    Args: 

    model -> Keras Model class model
    lr -> learning rate for the Adam optimizer
    loss -> loss metric to be used for training

    Returns: A compiled keras model
    '''
    model = model
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                   loss=loss)
    return model

def set_callback_and_early_stopping(patience: int = 10):
    """
    Callbacks for backing up checkpoints and early stopping for training

    Args: 
    
    backup_restore_dir -> Directory where model training checkpoints will be stored
    patience -> Number of epochs before training is automatically terminated if the val_loss
                does not decrease

    Returns:

    list of callback functions
    """
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience
    )
    return early_stopping

def train_model(model: Model, X_train, X_test, callbacks: list, save_name: str, save_path: str, num_epochs: int, batch_size: int, shuffle: bool=True, save_model: bool=True):
    '''
    Train the loaded keras model

    Args:

    model -> Keras Model for training
    X_train -> input training set
    X_test -> input test set
    callbacks -> list of callbacks
    save_path -> path to save model
    num_epochs -> number of training epochs
    batch_size -> batch size for training
    shuffle -> shuffle the dataset during training. Default: True
    save_model -> save model in .keras format after the training is completed. Default: True
    '''
    
    history = model.fit(X_train, X_train,
          epochs=num_epochs,
          batch_size=batch_size,
          validation_data=(X_test, X_test),
          callbacks=callbacks,
          shuffle=shuffle)
    if save_model:
        model.save(os.path.join(save_path, save_name+'.keras'))
        return history
    else:
        return history

def get_threshold(model, normal_train_data, factor: int=2) -> float:
    '''
    Generate error threshold for anomaly detection. 
    This will be used to detect anomalies in the decoder when a sequences is passed to the model

    Args:

    model -> Autoencoder model
    normal_train_data -> training data to obtain the MAE between GT and predictions
    factor -> Std Deviation multiplier for the threshold

    Returns threshold
    '''
    reconstructions = model.predict(normal_train_data)
    train_loss = tf.keras.losses.mae(reconstructions, normal_train_data)
    threshold = factor*(np.mean(train_loss) + np.std(train_loss))

    return float(threshold)