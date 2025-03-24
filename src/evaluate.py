import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


class AnomalyDetectionEvaluator:
    """
    Anomaly Detection Evaluator for reconstruction error analysis and performance metrics.
    """

    def __init__(self, threshold_factor: int = 2):
        """
        Initialize the evaluator with optional threshold factor.

        Args:
            threshold_factor (int): Multiplier for standard deviation to determine the threshold. Default is 2.
        """
        self.threshold_factor = threshold_factor

    def calculate_reconstruction_errors(self, model, data: np.ndarray) -> np.ndarray:
        """
        Calculate reconstruction errors for a dataset using an LSTM autoencoder model.

        Args:
            model: Trained LSTM autoencoder model.
            data (np.ndarray): Input data to test.

        Returns:
            np.ndarray: Reconstruction errors for each sample.
        """
        reconstructions = model.predict(data)
        errors = np.mean(np.square(data - reconstructions), axis=1)  # Mean Squared Error per sample
        return errors

    def determine_threshold(self, reconstruction_errors: np.ndarray) -> float:
        """
        Determine anomaly detection threshold based on reconstruction errors.

        Args:
            reconstruction_errors (np.ndarray): Reconstruction errors from normal training data.

        Returns:
            float: Threshold for anomaly detection.
        """
        mean_error = np.mean(reconstruction_errors)
        std_error = np.std(reconstruction_errors)
        return mean_error + self.threshold_factor * std_error

    def classify_anomalies(self, reconstruction_errors: np.ndarray, threshold: float) -> np.ndarray:
        """
        Classify samples as anomalies based on the reconstruction error threshold.

        Args:
            reconstruction_errors (np.ndarray): Reconstruction errors for samples.
            threshold (float): Threshold for anomaly detection.

        Returns:
            np.ndarray: Boolean array indicating anomalies (True = anomaly).
        """

        return reconstruction_errors > threshold

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """
        Calculate evaluation metrics for anomaly detection.

        Args:
            y_true (np.ndarray): Ground truth labels (1 for anomaly, 0 for normal).
            y_pred (np.ndarray): Predicted labels (True for anomaly, False for normal).

        Returns:
            dict: Dictionary of precision, recall, F1-score, and accuracy.
        """
        precision = precision_score(y_true, y_pred, zero_division=1)
        recall = recall_score(y_true, y_pred, zero_division=1)
        f1 = f1_score(y_true, y_pred, zero_division=1)
        accuracy = accuracy_score(y_true, y_pred)
        return {
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "Accuracy": accuracy
        }