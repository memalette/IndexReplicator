from abc import ABCMeta, abstractmethod


class Base(metaclass=ABCMeta):
    """Base class for agents using index_environment"""

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def learn(self):
        """method that trains agent and save best model params
        in '../models/' folder"""
        pass

    @abstractmethod
    def predict(self):
        """method that loads best model params and plots action,
        returns and values figures
        """
        pass

