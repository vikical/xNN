from operator import mod
import larq

from src.mappers.keras2larq import Keras2Larq

class LarqFactory:
    def __init__(self) -> None:
        pass

    @staticmethod
    def get_mapper(model):
        """
        Based on the library used to create the model, the function return its mapper
        """

        if "keras" in model.__module__:
            return Keras2Larq()

        return None