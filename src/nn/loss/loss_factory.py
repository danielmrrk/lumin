from src.nn.loss.cross_entropy import CrossEntropy
from src.nn.loss.mse import MSE
from src.nn.type import LossType

loss_factory = {
    LossType.MSE: MSE,
    LossType.CROSS_ENTROPY: CrossEntropy
}