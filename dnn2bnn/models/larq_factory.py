from operator import mod
import larq

from dnn2bnn.mappers.keras2larq import Keras2Larq

class LarqFactory:
    def __init__(self) -> None:
        pass

    @staticmethod
    def get_mapper(model, larq_configuration):
        """
        Based on the library used to create the model, the function return its mapper
        """

        if "keras" in model.__module__:
            return Keras2Larq(larq_configuration=larq_configuration)

        return None