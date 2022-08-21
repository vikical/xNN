import sklearn.metrics as skmetrics
import tensorflow as tf
import numpy as np

class Fidelity:
    def __init__(self,original, surrogate, x):
        self.original=original
        self.surrogate=surrogate
        self.x=x

    
    def accuracy(self, last_layer="softmax"):
        """
        Compute the accuracy between the prediction of the surrogate and the original model.
        If last_layer="softmax", the program assumes the model outputs are probabilities vectors and they must be transform into one-hot class vectors.
        If last_layer="sigmoid", the program assumes the model outputs one single probability value for each prediction.
        """
        original_predictions=self.original.predict(self.x)
        surrogate_predictions=self.surrogate.predict(self.x)

        if last_layer=="softmax":
            depth=original_predictions[0].size
            original_predictions=tf.one_hot(tf.argmax(original_predictions, axis = 1), depth = depth)
            surrogate_predictions=tf.one_hot(tf.argmax(surrogate_predictions, axis = 1), depth = depth)
       
        if last_layer=="sigmoid":
            original_predictions= np.array(tf.round(original_predictions)).flatten()
            surrogate_predictions= np.array(tf.round(surrogate_predictions)).flatten()

        return skmetrics.accuracy_score(original_predictions,surrogate_predictions)