from src.nn.normalization.max_norm import max_norm
from src.nn.type import NormalizationType

norm_factory = {
    NormalizationType.MAX: max_norm
}