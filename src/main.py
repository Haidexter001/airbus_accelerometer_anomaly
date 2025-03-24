
import numpy as np
import yaml
import pandas as pd
from data import DataFrameProcessor
from train import LSTMTrainer
from evaluate import AnomalyDetectionEvaluator
from model import LSTMModel
from utils import calculate_threshold

with open("../config.yaml", "r") as f:
    config = yaml.safe_load(f)

DF_TRAIN_PATH=config['input']['train_dataset']
DF_TEST_PATH=config['input']['test_dataset']
TRAINING_CONFIG = config['train']
CALLBACKS = config['callbacks']

LEARNING_RATE = TRAINING_CONFIG['learning_rate']
LOSS = TRAINING_CONFIG['loss']
OPTIMIZER = TRAINING_CONFIG['optimizer']
EPOCHS = TRAINING_CONFIG['epochs']
MODEL_SAVE_NAME = TRAINING_CONFIG['save_model_name']
BATCH_SIZE = TRAINING_CONFIG['batch_size']
CHECKPOINT_DIR = CALLBACKS['checkpoint_dir']
PATIENCE = CALLBACKS['patience']
MONITOR = CALLBACKS['monitor']

def load_dataset(train_df_path: str, test_df_path: str):
    train_df = pd.read_hdf(train_df_path)
    test_df = pd.read_hdf(test_df_path)
    data_processer = DataFrameProcessor()
    train_data = data_processer.reshape_data(train_df)
    test_data = data_processer.reshape_data(test_df)
    test_labels = data_processer.generate_test_labels(test_df)
    input_shape = (train_data.shape[1], train_data.shape[2])

    return train_data, test_data, input_shape, test_labels

def create_model(input_shape: tuple):
    anomaly_detector = LSTMModel(input_shape=input_shape, regularizer=0.01, optimizer=OPTIMIZER, loss=LOSS)
    anomaly_detector.compile_and_build()
    anomaly_detector.build_graph()
    anomaly_detector.summary()

    return anomaly_detector

def main():
    try:
        # Step 1: Prepare dataset
        print("Loading dataset...")
        train_data, test_data, input_shape, test_labels = load_dataset(DF_TRAIN_PATH, DF_TEST_PATH)
        print(f"Dataset loaded: {len(train_data)} training samples, {len(test_data)} test samples.")

        # Step 2: Load and train the model
        print("Initializing and training the LSTM model...")
        model = create_model(input_shape=input_shape)
        trainer = LSTMTrainer(patience=5, backup_restore_dir="../checkpoints/lstm_model.keras")

        history = trainer.train_model(
            model=model,
            X_train=train_data,
            X_test=train_data,
            save_name="lstm_anomaly_detector",
            save_path="../saved_models/",
            num_epochs=EPOCHS,
            batch_size=BATCH_SIZE
        )
        print("Model training completed.")

        # Step 3: Calculate the anomaly detection threshold
        print("Calculating anomaly detection threshold...")
        threshold = calculate_threshold(model=model, normal_train_data=train_data, factor=2)
        print(f"Anomaly detection threshold: {threshold}")

        # Step 4: Evaluate the model
        print("Evaluating the model...")
        evaluator = AnomalyDetectionEvaluator(threshold_factor=2)
        train_errors = evaluator.calculate_reconstruction_errors(model, train_data)
        test_errors = evaluator.calculate_reconstruction_errors(model, test_data)

        # Recalculate threshold using training errors
        threshold = evaluator.determine_threshold(train_errors)
        print(f"Recalculated threshold: {threshold}")

        # Classify test data based on reconstruction errors
        y_pred = evaluator.classify_anomalies(test_errors, threshold)
        y_pred_reduced = np.mean(y_pred, axis=1).astype(int)
        
        # Calculate and display metrics
        metrics = evaluator.calculate_metrics(y_true=test_labels, y_pred=y_pred_reduced)
        print("Evaluation Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
