from src.nn.optimizer.adam import Adam
from src.nn.optimizer.gradient_descent import GradientDescent
from src.nn.optimizer.optimizer import Optimizer
from src.nn.optimizer.rms_prop import RMSProp
from src.nn.optimizer.sgd import SGD
from src.nn.optimizer.sgd_momentum import SGDMomentum
from src.nn.type import OptimizerType

optimizer_factory = {
    OptimizerType.GRADIENT_DESCENT: GradientDescent,
    OptimizerType.SGD: SGD,
    OptimizerType.SGD_MOMENTUM: SGDMomentum,
    OptimizerType.RMS_PROP: RMSProp,
    OptimizerType.ADAM: Adam
}

def create_optimizer(optimizer_type: OptimizerType, **kwargs) -> Optimizer:
    try:
        return optimizer_factory[optimizer_type](**kwargs)
    except KeyError:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
