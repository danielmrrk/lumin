from src.nn.activation.relu import ReLu
from src.nn.type import ActivationType

activation_factory = {
    ActivationType.RELU: ReLu
}