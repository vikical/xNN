from pickle import TRUE
import tensorflow.keras as tfkeras
from dnn2bnn.models.larq_factory import LarqFactory

class ModelManager:

    
    def __init__(self, original_model, larq_configuration):
        self.original_model=original_model
        self.larq_configuration=larq_configuration

    def __create_layer(self, original_layer, ignore_input_quantization=False):
        """
        Creates the corresponding binarized (if possible) layer.
        """
        layer_mapper=LarqFactory.get_mapper(model=self.original_model, larq_configuration=self.larq_configuration)
        binarized=True

        #If mapper is available, we try to map the layer
        new_layer=None
        if layer_mapper is not None:
            new_layer=layer_mapper.create_larq_layer(original_layer, ignore_input_quantization)  

        #If we didn't get a new layer (because the there's no mapper available or the mapper cannot translate it),
        #we copy it.
        if new_layer is None:
            binarized=False
            new_layer=self.__clone_layer(original_layer=original_layer)

        #We build the new_layer (not matter whether the layer is directly cloned or translated into LARQ)        
        return (new_layer, binarized)

    def __clone_layer(self,original_layer):
        layer_config = original_layer.get_config()  
        new_layer = type(original_layer).from_config(layer_config)        
        return new_layer

    def create_larq_model(self, reset_weights=True):
        """
        Creates a LARQ model (binarized) based on one original model.
        If reset_weights parameter is set to True, the model is build and the weights of the original model are initialized.
        In case reset_weights is set to False, the weights are kept and used as seed for the surrogate training.
        """
        original_model=self.original_model
        larq_model=tfkeras.models.Sequential()

        #Add layers.
        binarized=False
        at_least_one_binarized=False

        #Add first layer.
        (input_layer,binarized)=self.__create_layer(original_layer=original_model.layers[0])        
        larq_model.add(input_layer)        

        #Add hidden layers.
        for i in range(1,len(original_model.layers)-1):
            layer_to_be_replicated=original_model.layers[i]
            (new_layer,binarized)=self.__create_layer(original_layer=layer_to_be_replicated, 
            ignore_input_quantization=(at_least_one_binarized==False))
            if at_least_one_binarized==False and binarized:
                at_least_one_binarized=True
            larq_model.add(new_layer)
        
        #Add last layer.
        new_layer=self.__clone_layer(original_layer=original_model.layers[len(original_model.layers)-1])
        larq_model.add(new_layer)

        #Build the model pointing out the input shape.
        if reset_weights:
            larq_model.build(input_shape=original_model.input_shape)

        #Assign by default the compilation options given by the programmer to the original_model.
        #NOTE: instead of copying the optimizer, we create a new one. 
        #Reason behind: if the original_model has been trained, its learning rate isn't appropriate for the new model.
        larq_model.compile(optimizer=original_model.optimizer.__class__.__name__,
            loss=original_model.compiled_loss._user_losses,
            loss_weights=original_model.compiled_loss._user_loss_weights,
            metrics=original_model.compiled_metrics._user_metrics,
            weighted_metrics=original_model.compiled_metrics._user_weighted_metrics,
            run_eagerly=original_model._run_eagerly)
            
        return larq_model
