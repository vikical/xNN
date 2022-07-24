from configparser import ConfigParser
from http.client import CONFLICT
import sys, inspect
from warnings import filters
import larq.layers
import tensorflow.keras as tfkeras



class Keras2Larq:

    def __init__(self, larq_configuration):
        self.binarizableLayers=["Dense","Conv1D","Conv2D","Conv3D","DepthwiseConv2D",
        "SeparableConv1D","SeparableConv2D","Conv2DTranspose","Conv3DTranspose",
        "LocallyConnected1D","LocallyConnected2D"]

        self.__CLASS_PREFIX="Quant"
        self.__MODULE_PREFIX="larq.layers"

        self.larq_configuration=larq_configuration

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
              
    def __instance_layer(self,original_layer,class_name,module_name):
        """
        Instance a layer, copying common paremeters from the original layer.
        No common paremeters has to be configured or passed through the main arguments.
        """
        class_type= getattr(sys.modules[module_name], class_name)
        parameters_info= inspect.getargspec(class_type.__init__)

        #We get the values from configuration and from equivalent class in tf.keras.
        values={}
        for param in parameters_info.args:
            if param=="self":
                continue

            # If the parameter has been indicated through the configuration file, we take its value.
            if param in self.larq_configuration:
                values[param]=self.larq_configuration.get(param)
                continue
        
            #If no indication has been passed, we get the value from the Keras equivalent.
            #If it doesn't exist, an Error is thrown.
            values[param]=getattr(original_layer, param)


        larq_layer= class_type(**values)

        return larq_layer              

    def __instance_larq_layer(self,original_layer,larq_classname):
        """
        Instance a LARQ layer, copying common paremeters with Keras layer.
        No common paremeters has to be configured or passed through the main arguments.
        """
        return self.__instance_layer(original_layer=original_layer,class_name=larq_classname,module_name=self.__MODULE_PREFIX)

    def create_larq_layer(self,original_layer, ignore_input_quantization=False):
        """
        Given a layer, returns None if it cannot be translated into LARQ or the corresponding Larq layer, otherwise.
        If ignore_input_quantization=False, input_quantization is set to None, even if the configuration has other value.
        """

        larq_classname=self._get_equivalent_larq_classname(layer=original_layer)

        if larq_classname is None:
            return None
        
        larq_layer=self.__instance_larq_layer(original_layer=original_layer,larq_classname=larq_classname)
        if ignore_input_quantization:
            larq_layer.input_quantizer=None

        return larq_layer


        