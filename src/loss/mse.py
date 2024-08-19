from .loss import Loss
from src.utility.loss_type import LossType


class MSE(Loss):
    def get_technical_name(self) -> LossType:
        return LossType.MSE