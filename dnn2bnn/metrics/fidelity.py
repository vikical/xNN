import sklearn.metrics as skmetrics
import tensorflow as tf

class Fidelity:
    def __init__(self,original, surrogate, x):
        self.original=original
        self.surrogate=surrogate
        self.x=x

    
    def accuracy(self, prob2class=True):
        """
        Compute the accuracy between the prediction of the surrogate and the original model.
        If prob2class=True, the program assumes the model outputs are probabilities vectors and they must be transform into one-hot class vectors.
        """
        original_predictions=self.original.predict(self.x)
        surrogate_predictions=self.surrogate.predict(self.x)

        if prob2class:
            depth=original_predictions[0].size
            original_predictions=tf.one_hot(tf.argmax(original_predictions, axis = 1), depth = depth)
            surrogate_predictions=tf.one_hot(tf.argmax(surrogate_predictions, axis = 1), depth = depth)
       
        return skmetrics.accuracy_score(original_predictions,surrogate_predictions)