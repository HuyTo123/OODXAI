# D:\Code\xAIandOOD\oodxai\__init__.py

# Điều này sẽ cho phép bạn import DeepExplainer trực tiếp từ oodxai
from .Explain._deep.__init__ import DeepExplainer # Giả sử DeepExplainer nằm trong deep_pytorch.py
# HOẶC nếu bạn muốn ẩn cấu trúc _deep, bạn có thể import từ Explain.__init__.py
# from .Explain import DeepExplainer # Nếu bạn muốn expose DeepExplainer qua Explain.__init__.py

# Bạn có thể thêm các import khác ở đây để dễ truy cập
from .OodDeepSHAP import OODDeepExplainer # Giả sử OodDeepSHAP là một class/hàm bạn muốn expose

__all__ = [
    "DeepExplainer", # Để khi from oodxai import * nó bao gồm DeepExplainer
    "OodDeepSHAP"
]