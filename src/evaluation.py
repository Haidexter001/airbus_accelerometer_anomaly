import tensorflow as tf
from tensorflow.keras.models import Model
from sklearn.metrics import accuracy_score, precision_score, recall_score

def _predict_test(model: Model, data, threshold):
  '''
  Uses model to predict the threshold

  Args:
  
  model -> Trained Keras model
  data -> Dataset to run predictions
  threshold -> initial threshold for predictions

  Returns MAE threshold
  '''
  reconstructions = model.predict(data)
  loss = tf.keras.losses.mae(reconstructions, data)
  return tf.math.less(loss, threshold)

def _print_stats(labels, predictions):
  '''
  Print Precision, Recall, and Accuracy of predictions for given labels

  Args:

  labels -> Ground truth labels
  predictions -> model prediction results
  '''
  print("Accuracy = {}".format(accuracy_score(labels, predictions)))
  print("Precision = {}".format(precision_score(labels, predictions)))
  print("Recall = {}".format(recall_score(labels, predictions)))


def show_precision_recall(model, test_data, threshold, test_labels):
  '''
  Display prediction results of model
  '''
  preds = _predict_test(model, test_data, threshold)
  _print_stats(preds, test_labels)