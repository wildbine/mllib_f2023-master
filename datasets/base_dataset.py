import numpy as np
from abc import ABC, abstractmethod
import random as rnd

class BaseDataset(ABC):

    def __init__(self,train_set_percent,valid_set_percent):
        self.train_set_percent = train_set_percent
        self.valid_set_percent = valid_set_percent

    @property
    @abstractmethod
    def targets(self):
        # targets variables
        pass

    @property
    @abstractmethod
    def inputs(self):
        # inputs variables
        pass


    def _divide_into_sets(self):
        data = list((zip(self.inputs, self.targets)))
        rnd.shuffle(data)
        n = len(data)
        training_count = int(self.train_set_percent * n)
        valid_count = int(self.valid_set_percent * n)
        training_set = data[:training_count]
        valid_set = data[training_count:training_count + valid_count]
        # распаковали уже разбитые на дизъюнктное объединение данные
        self.training_inputs, self.training_targets = zip(*training_set)
        self.valid_inputs, self.valid_targets = zip(*valid_set)
