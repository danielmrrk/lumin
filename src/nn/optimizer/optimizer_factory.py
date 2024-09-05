from src.nn.optimizer.gradient_descent import GradientDescent
from src.nn.optimizer.optimizer import Optimizer
from src.nn.type import OptimizerType

optimizer_factory = {
    OptimizerType.GRADIENT_DESCENT: GradientDescent
}

def create_optimizer(optimizer_type: OptimizerType, **kwargs) -> Optimizer:
    try:
        return optimizer_factory[optimizer_type](**kwargs)
    except KeyError:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
