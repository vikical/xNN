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
        obtained=my_layer.__class__.__name__
        self.assertEqual(expected,obtained,"binarized layer should belong to larq.layers")

    def test_instance_dense(self):
        original_layer = tf.keras.layers.Dense(units=3)
        bin_layer=self.k2l.create_larq_dense(original_layer=original_layer)
        expected="QuantDense"
        obtained=bin_layer.__class__.__name__
        self.assertEqual(expected,obtained,"A Dense layer should be translated into a QuantDense")
