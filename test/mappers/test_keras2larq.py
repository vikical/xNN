import unittest, larq
import tensorflow as tf

from src.mappers.keras2larq import Keras2Larq

class TestKeras2Larq(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.k2l= Keras2Larq()
        x = tf.keras.Input(shape=(28, 28, 1))  
        y = tf.keras.layers.Conv2D(32, 3, strides=2, padding="same")(x)
        self.model = tf.keras.Model(inputs=x, outputs=y)
        self.model.build(input_shape=(28, 28, 1))



    def test_Conv2D_equivalent_is_QuantConv2D(self):
        expected="QuantConv2D"
        obtained=self.k2l._get_equivalent_larq_classname(self.model.layers[1])
        self.assertEqual(expected, obtained, "if Conv2D the binarizable is QuantConv2D")

    def test_Input_equivalent_is_None(self):
        expected=None
        obtained=self.k2l._get_equivalent_larq_classname(self.model.layers[0])
        self.assertEqual(expected, obtained, "if Input the binarizable is None")

    def test_created_class_belongs_to_larq(self):
        expected="larq.layers"
        my_layer=self.k2l.create_larq_layer(self.model.layers[1])
        obtained=my_layer.__class__.__module__
        self.assertEqual(expected,obtained,"binarized layer should belong to larq.layers")

    def test_instance_dense(self):
        original_layer = tf.keras.layers.Dense(units=3)
        bin_layer=self.k2l.create_larq_layer(original_layer=original_layer)
        expected="QuantDense"
        obtained=bin_layer.__class__.__name__
        self.assertEqual(expected,obtained,"The layer should be translated into "+expected)

    def test_instance_conv1d(self):
        original_layer = tf.keras.layers.Conv1D(filters=32,kernel_size=3)
        bin_layer=self.k2l.create_larq_layer(original_layer=original_layer)
        expected="QuantConv1D"
        obtained=bin_layer.__class__.__name__
        self.assertEqual(expected,obtained,"The layer should be translated into "+expected)

    def test_instance_conv2d(self):
        original_layer = tf.keras.layers.Conv2D(2,3)
        bin_layer=self.k2l.create_larq_layer(original_layer=original_layer)
        expected="QuantConv2D"
        obtained=bin_layer.__class__.__name__
        self.assertEqual(expected,obtained,"The layer should be translated into "+expected)

    def test_instance_conv3d(self):
        original_layer = tf.keras.layers.Conv3D(2,3)
        bin_layer=self.k2l.create_larq_layer(original_layer=original_layer)
        expected="QuantConv3D"
        obtained=bin_layer.__class__.__name__
        self.assertEqual(expected,obtained,"The layer should be translated into "+expected)

    def test_instance_depthwiseconv2d(self):
        original_layer = tf.keras.layers.DepthwiseConv2D(kernel_size=3)
        bin_layer=self.k2l.create_larq_layer(original_layer=original_layer)
        expected="QuantDepthwiseConv2D"
        obtained=bin_layer.__class__.__name__
        self.assertEqual(expected,obtained,"The layer should be translated into "+expected)

    def test_instance_separableconv1d(self):
        original_layer = tf.keras.layers.SeparableConv1D(filters=32,kernel_size=3)
        bin_layer=self.k2l.create_larq_layer(original_layer=original_layer)
        expected="QuantSeparableConv1D"
        obtained=bin_layer.__class__.__name__
        self.assertEqual(expected,obtained,"The layer should be translated into "+expected)

    def test_instance_separableconv2d(self):
        original_layer = tf.keras.layers.SeparableConv2D(filters=32,kernel_size=3)
        bin_layer=self.k2l.create_larq_layer(original_layer=original_layer)
        expected="QuantSeparableConv2D"
        obtained=bin_layer.__class__.__name__
        self.assertEqual(expected,obtained,"The layer should be translated into "+expected)

    def test_instance_conv2dtranspose(self):
        original_layer = tf.keras.layers.Conv2DTranspose(filters=32,kernel_size=3)
        bin_layer=self.k2l.create_larq_layer(original_layer=original_layer)
        expected="QuantConv2DTranspose"
        obtained=bin_layer.__class__.__name__
        self.assertEqual(expected,obtained,"The layer should be translated into "+expected)

    def test_instance_conv3dtranspose(self):
        original_layer = tf.keras.layers.Conv3DTranspose(filters=32,kernel_size=3)
        bin_layer=self.k2l.create_larq_layer(original_layer=original_layer)
        expected="QuantConv3DTranspose"
        obtained=bin_layer.__class__.__name__
        self.assertEqual(expected,obtained,"The layer should be translated into "+expected)

    def test_instance_locallyconnected1d(self):
        original_layer = tf.keras.layers.LocallyConnected1D(filters=32,kernel_size=3)
        bin_layer=self.k2l.create_larq_layer(original_layer=original_layer)
        expected="QuantLocallyConnected1D"
        obtained=bin_layer.__class__.__name__
        self.assertEqual(expected,obtained,"The layer should be translated into "+expected)

    def test_instance_locallyconnected2d(self):
        original_layer = tf.keras.layers.LocallyConnected2D(filters=32,kernel_size=3)
        bin_layer=self.k2l.create_larq_layer(original_layer=original_layer)
        expected="QuantLocallyConnected2D"
        obtained=bin_layer.__class__.__name__
        self.assertEqual(expected,obtained,"The layer should be translated into "+expected)
