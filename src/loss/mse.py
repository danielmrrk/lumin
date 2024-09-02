from .loss import Loss
from src.utility.type import LossType


class MSE(Loss):
    def get_technical_name(self) -> LossType:
        return LossType.MSE