import unittest, larq
import tensorflow as tf

from src.models.larq_factory import LarqFactory

class TestLarqFactory(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.lf= LarqFactory()

    def test_prueba(self):
        pass