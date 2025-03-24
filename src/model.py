import tensorflow as tf
from keras.layers import Dense, LSTM, TimeDistributed, RepeatVector
from keras.models import Model
from keras import regularizers

class LSTMModel(Model):
    """
    A custom LSTM autoencoder model inheriting from tf.keras.Model.
    """

    def __init__(self, input_shape: tuple, regularizer: float = 0.01, optimizer: str = "adam", loss: str = "mse"):
        """
        Initialize the model with required parameters.

        Args:
            input_shape (tuple): Shape of the input data (timesteps, features).
            regularizer (float): L2 regularization factor for the LSTM layers. Default is 0.01.
            optimizer (str): Optimizer for model compilation. Default is 'adam'.
            loss (str): Loss function for model compilation. Default is 'mse' (mean squared error).
        """
        super(LSTMModel, self).__init__()
        self.input_shape = input_shape
        self.regularizer = regularizer

        # Define the layers
        self.encoder_lstm1 = LSTM(160, activation='relu', return_sequences=True,
                                  kernel_regularizer=regularizers.l2(self.regularizer))
        self.encoder_lstm2 = LSTM(40, activation='relu', return_sequences=False)
        self.repeat_vector = RepeatVector(self.input_shape[0])
        self.decoder_lstm1 = LSTM(40, activation='relu', return_sequences=True)
        self.decoder_lstm2 = LSTM(160, activation='relu', return_sequences=True)
        self.output_layer = TimeDistributed(Dense(self.input_shape[1]))

        # Compile settings
        self.optimizer = optimizer
        self.loss = loss

    def call(self, inputs, training=None):
        """
        Forward pass of the model.

        Args:
            inputs (tensor): Input tensor for the model.
            training (bool, optional): Whether the model is in training mode.

        Returns:
            tensor: Output tensor after passing through the layers.
        """
        # Encoder
        x = self.encoder_lstm1(inputs)
        x = self.encoder_lstm2(x)

        # Latent space and Decoder
        x = self.repeat_vector(x)
        x = self.decoder_lstm1(x)
        x = self.decoder_lstm2(x)
        outputs = self.output_layer(x)

        return outputs

    def compile_and_build(self):
        """
        Compile the model with the specified optimizer and loss.
        """
        self.compile(optimizer=self.optimizer, loss=self.loss)
        self.build(input_shape=(None, *self.input_shape))

    def build_graph(self):
        """
        Build the computation graph explicitly for model.summary().
        """
        dummy_input = tf.keras.Input(shape=self.input_shape)  # Create a dummy input tensor
        self.call(dummy_input)  # Call the model with the dummy input