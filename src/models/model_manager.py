import tensorflow.keras as tfkeras
from src.models.larq_factory import LarqFactory

class ModelManager:

    
    def __init__(self, original_model):
        self.original_model=original_model

    def __create_layer(self, original_layer, ignore_input_quantization=False):
        """
        Creates the corresponding binarized (if possible) layer.
        """
        layer_mapper=LarqFactory.get_mapper(self.original_model)
        binarized=True

        #If mapper is available, we try to map the layer
        new_layer=None
        if layer_mapper is not None:
            new_layer=layer_mapper.create_larq_layer(original_layer, ignore_input_quantization)                        

        #If we didn't get a new layer (because the there's no mapper available o the mapper cannot translate it),
        #we copy it.
        if new_layer is None:
            binarized=False
            layer_config = original_layer.get_config()  
            new_layer = type(original_layer).from_config(layer_config)
            new_layer.build(original_layer.input_shape)

        return (new_layer, binarized)


    def create_larq_model(self,original_model):
        """
        Creates a LARQ model (binarized) based on one original model.
        """
        preprocessing_step_is_over=False
        larq_model=tfkeras.models.Sequential()

        #Add layers.
        binarized=False
        (input_layer,binarized)=self.__create_layer(original_model.layers[0])        
        larq_model.add(input_layer)        

        for i in range(1,len(original_model.layers)):
            layer_to_be_replicated=original_model.layers[i]
            (new_layer,binarized)=self.__create_layer(original_layer=layer_to_be_replicated,
            ignore_input_quantization=(not binarized))
            larq_model.add(new_layer)
        
        #Build the model pointing out the input shape.
        larq_model.build(input_shape=original_model.input_shape)

        #Assign by default the compilation options given by the programmer to the original_model.
        larq_model.compile(optimizer=original_model.optimizer,
            loss=original_model.compiled_loss._user_losses,
            loss_weights=original_model.compiled_loss._user_loss_weights,
            metrics=original_model.compiled_metrics._user_metrics,
            weighted_metrics=original_model.compiled_metrics._user_weighted_metrics,
            run_eagerly=original_model._run_eagerly)

        return larq_model
