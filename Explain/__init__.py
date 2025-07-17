from ._deep import DeepExplainer
# from ._gradient import GradientExplainer
from ._kernel import KernelExplainer

Deep = DeepExplainer
Kernel = KernelExplainer
# Gradient = GradientExplainer

__all__ = [
    "DeepExplainer",
    "GradientExplainer",
    "KernelExplainer",
]