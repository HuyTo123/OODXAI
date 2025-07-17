# D:\Code\xAIandOOD\oodxai\__init__.py

# Điều này sẽ cho phép bạn import DeepExplainer trực tiếp từ oodxai
from .Explain._deep.__init__ import DeepExplainer # Giả sử DeepExplainer nằm trong deep_pytorch.py
from .Explain._kernel import KernelExplainer # Giả sử KernelExplainer nằm trong kernel_pytorch.py

# Bạn có thể thêm các import khác ở đây để dễ truy cập
from .Explain.main.OodDeepSHAP import OODDeepExplainer # Giả sử OodDeepSHAP là một class/hàm bạn muốn expose

__all__ = [
    "DeepExplainer", # Để khi from oodxai import * nó bao gồm DeepExplainer
    "OodDeepSHAP"
    "KernelExplainer", # Để khi from oodxai import * nó bao gồm KernelExplainer
]