import sys, inspect
from warnings import filters
import larq.layers
import tensorflow.keras as tfkeras


class Keras2Larq:

    def __init__(self):
        self.binarizableLayers=["Dense","Conv1D","Conv2D","Conv3D","DepthwiseConv2D",
        "SeparableConv1D","SeparableConv2D","Conv2DTranspose","Conv3DTranspose",
        "LocallyConnected1D","LocallyConnected2D"]

        self.__CLASS_PREFIX="Quant"
        self.__MODULE_PREFIX="larq.layers"

        return

    def _get_equivalent_larq_classname(self,layer):
        '''
        Get the equivalent larq layer class name only if the original layer belongs to keras library.
        '''
        #Check whether it belongs to keras.
        module_prefix:str=layer.__class__.__module__
        if 'keras' not in module_prefix:
            return None

        #We get the keras class name.
        keras_classname=layer.__class__.__name__

        #If the layer is binarizable we translate it, otherwise we translate it.
        if keras_classname not in self.binarizableLayers:
            return None

        return self.__CLASS_PREFIX+keras_classname
              
    def __instance_larq_layer(self,original_layer,larq_classname):
        """
        Instance a LARQ layer, copying common paremeters with Keras layer.
        No common paremeters has to be configured or passed through the main arguments.
        """
        module_name=self.__MODULE_PREFIX
        class_type= getattr(sys.modules[module_name], larq_classname)
        parameters_info= inspect.getargspec(class_type.__init__)

        #TODO: get quantizer parameters from config file or program input.
        config={"pad_values":0.0,
        "input_quantizer":None,
        "depthwise_quantizer":None,
        "pointwise_quantizer":None,
        "kernel_quantizer":None}


        #We get the values from configuration and from equivalent class in tf.keras.
        values={}
        for param in parameters_info.args:
            if param=="self":
                continue

            # If the parameter has been indicated manually, we take its value.
            if param in config:
                values[param]=config.get(param)
                continue
        
            #If no indication has been passed, we get the value from the Keras equivalent.
            #If it doesn't exist, an Error is thrown.s
            values[param]=getattr(original_layer, param)


        larq_layer= class_type(**values)

        return larq_layer



    def create_larq_layer(self,original_layer):
        """
        Given a layer, returns None if it cannot be translated into LARQ or the corresponding Larq layer, otherwise.
        """

        larq_classname=self._get_equivalent_larq_classname(layer=original_layer)

        if larq_classname is None:
            return None
        
        return self.__instance_larq_layer(original_layer=original_layer,larq_classname=larq_classname)
