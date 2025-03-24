import os
import tensorflow as tf
from tensorflow.keras.models import Model

class LSTMTrainer:
    """
    A class to handle training, callbacks, and threshold generation for an LSTM model.
    """

    def __init__(self, patience: int = 10, backup_restore_dir: str = None):
        """
        Initialize the LSTMTrainer class with optional callback parameters.

        Args:
            patience (int): Number of epochs for early stopping. Default is 10.
            backup_restore_dir (str): Directory to save model checkpoints. Default is None.
        """
        self.patience = patience
        self.backup_restore_dir = backup_restore_dir
        self.callbacks = self._set_callbacks()

    def _set_callbacks(self):
        """
        Create callback functions for early stopping and optional checkpointing.

        Returns:
            list: List of callbacks.
        """
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=self.patience)]
        if self.backup_restore_dir:
            checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=self.backup_restore_dir,
                save_best_only=True,
                monitor="val_loss"
            )
            callbacks.append(checkpoint_callback)
        return callbacks

    def train_model(self, model: Model, X_train, X_test, save_name: str, save_path: str, 
                    num_epochs: int, batch_size: int, shuffle: bool = True, save_model: bool = True):
        """
        Train the LSTM model with the specified parameters.

        Args:
            model (Model): Keras Model for training.
            X_train: Input training set.
            X_test: Input test set.
            save_name (str): File name for saving the model.
            save_path (str): Directory to save the model.
            num_epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            shuffle (bool): Whether to shuffle the dataset during training. Default is True.
            save_model (bool): Whether to save the model after training. Default is True.

        Returns:
            history: Training history object.
        """
        # Train the model
        history = model.fit(
            X_train, X_train,  # Assuming autoencoder; modify for supervised learning
            epochs=num_epochs,
            batch_size=batch_size,
            validation_data=(X_test, X_test),
            callbacks=self.callbacks,
            shuffle=shuffle
        )

        # Save the model if required
        if save_model:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            model.save(os.path.join(save_path, save_name + ".keras"))

        return history