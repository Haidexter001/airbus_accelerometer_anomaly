
import numpy as np
import yaml
from data_prep import extract_raw_data, generate_train_test_split
from trainer import compile_model, set_callback_and_early_stopping, train_model, get_threshold
from evaluation import show_precision_recall
from model import AnomalyDetector

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

DF_NORMAL_PATH=config['input']['normal_dataset']
DF_ABNORMAL_PATH=config['input']['abnormal_dataset']
TRAINING_CONFIG = config['train']
CALLBACKS = config['callbacks']

LEARNING_RATE = TRAINING_CONFIG['learning_rate']
LOSS = TRAINING_CONFIG['loss']
EPOCHS = TRAINING_CONFIG['epochs']
MODEL_SAVE_NAME = TRAINING_CONFIG['save_model_name']
TEST_TRAIN_SPLIT = TRAINING_CONFIG['val_split']
RANDOM_SEED = TRAINING_CONFIG['seed']
BATCH_SIZE = TRAINING_CONFIG['batch_size']
CHECKPOINT_DIR = CALLBACKS['checkpoint_dir']
PATIENCE = CALLBACKS['patience']
MONITOR = CALLBACKS['monitor']

def main():
    # Prepare dataset
    raw_data = extract_raw_data(DF_NORMAL_PATH, DF_ABNORMAL_PATH)
    normal_train_data, anomalous_train_data, test_data, anomalous_test_data, test_labels = generate_train_test_split(
        raw_data, test_size=TEST_TRAIN_SPLIT, random_state=RANDOM_SEED
    )

    callbacks = set_callback_and_early_stopping(PATIENCE)

    # Load model
    autoencoder= AnomalyDetector()
    autoencoder = compile_model(autoencoder, LEARNING_RATE, LOSS)
    history = train_model(autoencoder, normal_train_data, test_data, 
                          callbacks=callbacks, save_name=MODEL_SAVE_NAME, save_path=CHECKPOINT_DIR,
                          num_epochs=EPOCHS, batch_size=BATCH_SIZE)    
    
    # Evaluate model
    thresh = get_threshold(autoencoder, normal_train_data=normal_train_data, factor=2)
    show_precision_recall(autoencoder,test_data=test_data, test_labels=test_labels, threshold=thresh)

if __name__ == "__main__":
    main()
