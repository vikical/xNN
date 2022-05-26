import sys
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
              

    def create_larq_dense(self,original_layer:tfkeras.layers.Dense):        
        layer=larq.layers.QuantDense(units=original_layer.units,
        activation=original_layer.activation,
        use_bias=original_layer.use_bias,
        kernel_initializer=original_layer.kernel_initializer,
        bias_initializer=original_layer.bias_initializer,
        kernel_regularizer=original_layer.kernel_regularizer,
        bias_regularizer=original_layer.bias_regularizer,
        activity_regularizer=original_layer.activity_regularizer,
        kernel_constraint=original_layer.kernel_constraint,
        bias_constraint=original_layer.bias_constraint,
        input_quantizer=None,
        kernel_quantizer=None)
        return layer

    def create_larq_conv1d(self,original_layer:tfkeras.layers.Conv1D):
        layer=larq.layers.QuantConv1D(filters=original_layer.filters,
        kernel_size=original_layer.kernel_size,
        strides=original_layer.strides,
        padding=original_layer.padding,
        data_format=original_layer.data_format,
        dilation_rate=original_layer.dilation_rate,
        groups=original_layer.groups,
        activation=original_layer.activation,
        use_bias=original_layer.use_bias,
        kernel_initializer=original_layer.kernel_initializer,
        bias_initializer=original_layer.bias_initializer,
        kernel_regularizer=original_layer.kernel_regularizer,
        bias_regularizer=original_layer.bias_regularizer,
        activity_regularizer=original_layer.activity_regularizer,
        kernel_constraint=original_layer.kernel_constraint,
        bias_constraint=original_layer.bias_constraint,
        pad_values=0.0,
        input_quantizer=None,
        kernel_quantizer=None)
        return layer

    def create_larq_conv2d(self,original_layer:tfkeras.layers.Conv2D):
        layer=larq.layers.QuantConv2D(filters=original_layer.filters,
        kernel_size=original_layer.kernel_size,
        strides=original_layer.strides,
        padding=original_layer.padding,        
        data_format=original_layer.data_format,
        dilation_rate=original_layer.dilation_rate,
        groups=original_layer.groups,
        activation=original_layer.activation,
        use_bias=original_layer.use_bias,
        kernel_initializer=original_layer.kernel_initializer,
        bias_initializer=original_layer.bias_initializer,
        kernel_regularizer=original_layer.kernel_regularizer,
        bias_regularizer=original_layer.bias_regularizer,
        activity_regularizer=original_layer.activity_regularizer,
        kernel_constraint=original_layer.kernel_constraint,
        bias_constraint=original_layer.bias_constraint,
        pad_values= 0.0,
        input_quantizer=None,
        kernel_quantizer=None)
        return layer

    def create_larq_conv3d(self,original_layer:tfkeras.layers.Conv3D):
        layer=larq.layers.QuantConv3D(filters=original_layer.filters,
        kernel_size=original_layer.kernel_size,
        strides=original_layer.strides,
        padding=original_layer.padding,
        data_format=original_layer.data_format,
        dilation_rate=original_layer.dilation_rate,
        groups=original_layer.groups,
        activation=original_layer.activation,
        use_bias=original_layer.use_bias,
        kernel_initializer=original_layer.kernel_initializer,
        bias_initializer=original_layer.bias_initializer,
        kernel_regularizer=original_layer.kernel_regularizer,
        bias_regularizer=original_layer.bias_regularizer,
        activity_regularizer=original_layer.activity_regularizer,
        kernel_constraint=original_layer.kernel_constraint,
        pad_values=0.0,
        input_quantizer=None,
        kernel_quantizer=None
        )
        return layer

    def create_larq_depthwiseconv2d(self,original_layer:tfkeras.layers.DepthwiseConv2D):
        layer=larq.layers.QuantDepthwiseConv2D(kernel_size=original_layer.kernel_size,
        strides=original_layer.strides,
        padding=original_layer.padding,
        depth_multiplier=original_layer.depth_multiplier,
        data_format=original_layer.data_format,
        dilation_rate=original_layer.dilation_rate,
        activation=original_layer.activation,
        use_bias=original_layer.use_bias,
        depthwise_initializer=original_layer.depthwise_initializer,
        bias_initializer=original_layer.bias_initializer,
        depthwise_regularizer=original_layer.depthwise_regularizer,
        bias_regularizer=original_layer.bias_regulizer,
        activity_regularizer=original_layer.activity_regularizer,
        depthwise_constraint=original_layer.depthwise_constraint,
        bias_constraint=original_layer.bias_constraint,
        pad_values=0.0,
        input_quantizer=None,
        depthwise_quantizer=None)
        return layer

    def create_larq_separableconv1d(self,original_layer: tfkeras.layers.SeparableConv1D):
        layer=larq.layers.QuantSeparableConv1D(filters=original_layer.filters,
        kernel_size=original_layer.kernel_size,
        strides=original_layer.strides,
        padding=original_layer.padding,
        data_format=original_layer.data_format,
        dilation_rate=original_layer.dilation_rate,
        depth_multiplier=original_layer.depth_multiplier,
        activation=original_layer.activation,
        use_bias=original_layer.use_bias,
        depthwise_initializer=original_layer.depthwise_initializer,
        pointwise_initializer=original_layer.pointwise_initializer,
        bias_initializer=original_layer.bias_initializer,
        depthwise_regularizer=original_layer.depthwise_regularizer,
        pointwise_regularizer=original_layer.pointwise_regularizer,
        bias_regularizer=original_layer.bias_regularizer,
        activity_regularizer=original_layer.activity_regularizer,
        depthwise_constraint=original_layer.depthwise_constraint,
        pointwise_constraint=original_layer.pointwise_constraint,
        bias_constraint=original_layer.bias_constraint,
        pad_values=0.0,
        input_quantizer=None,
        depthwise_quantizer=None,
        pointwise_quantizer=None)
        return layer

    def create_larq_separableconv2d(self,original_layer:tfkeras.layers.SeparableConv2D):
        layer=larq.layers.QuantSeparableConv2D(filters=original_layer.filters,
        kernel_size=original_layer.kernel_size,
        strides=original_layer.strides,
        padding=original_layer.padding,
        data_format=original_layer.data_format,
        dilation_rate=original_layer.dilation_rate,
        depth_multiplier=original_layer.depth_multiplier,
        activation=original_layer.activation,
        use_bias=original_layer.use_bias,
        depthwise_initializer=original_layer.depthwise_initializer,
        pointwise_initializer=original_layer.pointwise_initializer,
        bias_initializer=original_layer.bias_initializer,
        depthwise_regularizer=original_layer.depthwise_regularizer,
        pointwise_regularizer=original_layer.pointwise_regularizer,
        bias_regularizer=original_layer.bias_regularizer,
        activity_regularizer=original_layer.activity_regularizer,
        depthwise_constraint=original_layer.depthwise_constraint,
        pointwise_constraint=original_layer.pointwise_constraint,
        bias_constraint=original_layer.bias_constraint,
        pad_values=0.0,
        input_quantizer=None,
        depthwise_quantizer=None,
        pointwise_quantizer=None)
        return layer

    def create_larq_conv2dtranspose(self,original_layer:tfkeras.layers.Conv2DTranspose):
        layer=larq.layers.QuantConv2DTranspose(filters=original_layer.filters,
        kernel_size=original_layer.kernel_size,
        strides=original_layer.strides,
        padding=original_layer.padding,
        output_padding=original_layer.output_padding,
        data_format=original_layer.data_format,
        dilation_rate=original_layer.dilation_rate,
        activation=original_layer.activation,
        use_bias=original_layer.use_bias,
        kernel_initializer=original_layer.kernel_initializer,
        bias_initializer=original_layer.bias_initializer,
        kernel_regularizer=original_layer.kernel_regularizer,
        bias_regularizer=original_layer.bias_regularizer,
        activity_regularizer=original_layer.activity_regularizer,
        kernel_constraint=original_layer.kernel_constraint,        
        bias_constraint=original_layer.bias_constraint,
        input_quantizer=None,
        kernel_quantizer=None)
        return layer

    def create_larq_conv3dtranspose(self,original_layer:tfkeras.layers.Conv3DTranspose):
        layer=larq.layers.QuantConv3DTranspose(filters=original_layer.filters,
        kernel_size=original_layer.kernel_size,
        strides=original_layer.strides,
        padding=original_layer.padding,
        output_padding=original_layer.output_padding,
        data_format=original_layer.data_format,
        dilation_rate=original_layer.dilation_rate,
        activation=original_layer.activation,
        use_bias=original_layer.use_bias,
        kernel_initializer=original_layer.kernel_initializer,
        bias_initializer=original_layer.bias_initializer,
        kernel_regularizer=original_layer.kernel_regularizer,
        bias_regularizer=original_layer.bias_regularizer,
        activity_regularizer=original_layer.activity_regularizer,
        kernel_constraint=original_layer.kernel_constraint,
        bias_constraint=original_layer.bias_constraint,
        input_quantizer=None,
        kernel_quantizer=None)
        return layer

    def create_larq_locallyconnected1d(self,original_layer:tfkeras.layers.LocallyConnected1D):
        layer=larq.layers.QuantLocallyConnected1D(filters=original_layer.filters,
        kernel_size=original_layer.kernel_size,
        strides=original_layer.strides,
        padding=original_layer.padding,
        data_format=original_layer.data_format,
        activation=original_layer.activation,
        use_bias=original_layer.use_bias,
        kernel_initializer=original_layer.kernel_initializer,
        bias_initializer=original_layer.bias_initializer,
        kernel_regularizer=original_layer.kernel_regularizer,
        bias_regularizer=original_layer.bias_regularizer,
        activity_regularizer=original_layer.activity_regularizer,
        kernel_constraint=original_layer.kernel_constraint,
        bias_constraint=original_layer.bias_constraint,
        implementation=original_layer.implementation,
        input_quantizer=None,
        kernel_quantizer=None)
        return layer

    def create_larq_locallyconnected2d(self,original_layer:tfkeras.layers.LocallyConnected2D):
        layer=larq.layers.QuantLocallyConnected2D(filters=original_layer.filters,
        kernel_size=original_layer.kernel_size,
        strides=original_layer.strides,
        padding=original_layer.padding,
        data_format=original_layer.data_format,
        activation=original_layer.activation,
        use_bias=original_layer.use_bias,
        kernel_initializer=original_layer.kernel_initializer,
        bias_initializer=original_layer.bias_initializer,
        kernel_regularizer=original_layer.kernel_regularizer,
        bias_regularizer=original_layer.bias_regularizer,
        activity_regularizer=original_layer.activity_regularizer,
        kernel_constraint=original_layer.kernel_constraint,
        bias_constraint=original_layer.bias_constraint,
        implementation=original_layer.implementation,
        input_quantizer=None,
        kernel_quantizer=None,)
        return layer

    def create_larq_layer(self,original_layer):
        larq_classname=self._get_equivalent_larq_classname(layer=original_layer)

        module_name=self.__MODULE_PREFIX
        return getattr(sys.modules[module_name], larq_classname)
