from abc import ABCMeta, abstractmethod


class Base(metaclass=ABCMeta):
    """Base class for agents using index_environment"""

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def learn(self):
        """method that trains agent and save best model params
        in '../models/' folder
        Input: can receive in input environment or not
        Output: specific to model
        """
        pass

    @abstractmethod
    def predict(self, env, start=None, save=False, model_path='../models/ppo/', pred_id=None):
        """method that loads best model params and plots action,
        returns and values figures
        Input:
            env: environment used to predict
            start: if you want to specify a start day
            save: save figures or nor
            model_path: path where best model is saved
            pred_id: id for name file of figures
        Output: specific to model
        """
        pass

