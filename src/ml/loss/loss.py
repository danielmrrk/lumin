from abc import ABC, abstractmethod
from src.utility.type import LossType


class Loss(ABC):
    def __init__(self):
        self.technical_name: LossType = self.get_technical_name()

    @abstractmethod
    def get_technical_name(self) -> LossType:
        pass
