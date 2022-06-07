
import unittest, larq
import tensorflow as tf

#import src
from src.models.model_manager import ModelManager

class TestModelManager(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

        x = tf.keras.Input(shape=(28, 28, 1))  
        y = tf.keras.layers.Conv2D(32, 3, strides=2, padding="same")(x)
        self.model = tf.keras.Model(inputs=x, outputs=y)
        self.model.build(input_shape=(28, 28, 1))


    def test_mapped_model_has_as_many_layers_as_sequential_one(self):        
        original_model=tf.keras.models.load_model(filepath="./test/resources/sequential_mlp_2h.h5")
        mm=ModelManager(original_model=original_model)
        larq_model=mm.create_larq_model(original_model=original_model)
        expected=len(original_model.layers)
        obtained=len(larq_model.layers)
        self.assertEqual(expected,obtained,"The original model has "+ str(expected)+" but the mapped one has "+str(obtained))

    def test_mapped_model_has_one_layer_less_than_functional_one(self):        
        original_model=tf.keras.models.load_model(filepath="./test/resources/functional_mlp_2h.h5")
        mm=ModelManager(original_model=original_model)
        larq_model=mm.create_larq_model(original_model=original_model)
        expected=len(original_model.layers)-1
        obtained=len(larq_model.layers)
        self.assertEqual(expected,obtained,"The original model has "+ str(expected)+" but the mapped one has "+str(obtained))