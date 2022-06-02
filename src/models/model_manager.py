import tensorflow.keras as tfkeras
from src.models.larq_factory import LarqFactory

class ModelManager:

    
    def __init__(self, original_model):
        self.original_model=original_model

    def __create_layer(self, original_layer):
        layer_mapper=LarqFactory.get_mapper(self.original_model)

        #If mapper is available, we try to map the layer
        new_layer=None
        if layer_mapper is not None:
            new_layer=layer_mapper.create_larq_layer(original_layer)            

        #If we didn't get a new layer (because the there's no mapper available o the mapper cannot translate it),
        #we copy it.
        if new_layer is None:
            layer_config = original_layer.get_config()  
            new_layer = type(original_layer).from_config(layer_config)
            new_layer.build(original_layer.input_shape)

        return new_layer 


    def create_larq_model(self,original_model):
        larq_model=tfkeras.models.Sequential()

        input_layer=self.__create_layer(original_model.layers[0])
        larq_model.add(input_layer)

        for i in range(1,len(original_model.layers)):
            layer_to_be_replicated=original_model.layers[i]
            new_layer=self.__create_layer(layer_to_be_replicated)
            larq_model.add(new_layer)
        
        larq_model.build(input_shape=original_model.input_shape)

        return larq_model
