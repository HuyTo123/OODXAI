""" This method is combine from DeepSHAP and MSP methods."""
import numpy as np

from pytorch_ood.detector import MaxSoftmax, EnergyBased, ODIN, Mahalanobis, MCD
from matplotlib import colors
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from .. import DeepExplainer  
from .. import KernelExplainer
from scipy.stats import percentileofscore
from scipy.ndimage import gaussian_filter

from .plot import Plot

class OODDeepExplainer(Plot):
    def __init__(self, model=None, method_name=None, backgroundata=None, sample=None, device=None, xAI_method='DeepSHAP' ):
        """
        Initialize the DeepExplainer with MaxSoftmax for OOD detection Explanation.

        Parameters
        ----------
        model : torch.nn.Module
            The PyTorch model to explain.
        method : pytorch_ood_detector package
            The method to use for OOD detection, e.g., 'MSP'.
        backgroundata : None or np.ndarray, optional
            Data to use for initializing the explainer. If None, it will be inferred from the model.
        sample : None or torch.Tensor, optional
            Sample data to explain. If None, it will be inferred from the model.
        device : str or torch.device, optional
            Device to run the model on (e.g., 'cpu' or 'cuda').
        in_scores_for_calibration: 
            Use for calibration scores
        """
        self.model = model
        self.backgroundata = backgroundata
        self.device = device
        self.sample = sample
        self.ind_scores_for_calibration = None
        self.sample_scores = None
        self.ood_percentile = None
        self.probs = None
        self.shap_values = None
        self.visualization = Plot()
        self.xAI_method = xAI_method
        if self.xAI_method == 'DeepSHAP':
            if method_name == 'MSP':
                self.detector = MaxSoftmax(self.model)
            elif method_name == 'Energy':
                self.detector = EnergyBased(self.model)
            else:
                raise ValueError(f"The method '{method_name}' is not supported.")
            self.ind_scores_for_calibration = self.detector.predict(self.backgroundata).detach().numpy()
            print(f"Hiệu chỉnh hoàn tất trên {len(self.ind_scores_for_calibration)} mẫu.")
        elif self.xAI_method == 'KernalSHAP':
            pass
        else:
            raise ValueError(f"The xAI method '{self.xAI_method}' is not supported. Please use 'DeepSHAP' or 'KernalSHAP'.")
        with torch.no_grad():
            logits = self.model(self.sample)
            self.probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    def OOD_DeepSHAP(self):
        # DeepSHAP
        print('Bắt đầu tính toán Deep SHAP...')
        self.xAI_method = 'DeepSHAP'
        explainer = DeepExplainer(self.model, self.backgroundata)
        shap_values_raw = explainer.shap_values(self.sample, check_additivity=True)  # Chuyển đổi kích thước tensor về (1,3,128,128) để phù hợp với đầu vào của model
        
        # Xử lý shap_values để có dạng list cho hàm plot
        num_classes = shap_values_raw.shape[-1]
        self.shap_values = [shap_values_raw[0, :, :, :, i] for i in range(num_classes)]
        print('Đã tính toán và xử lý xong SHAP values.')
        # MaxSoftmax
        self.sample_scores = self.detector(self.sample).item()
        print('Đã tính toán điểm số MaxSoftmax cho mẫu.')
        self.ood_percentile = percentileofscore(self.ind_scores_for_calibration, self.sample_scores)
        print('Đã tính toán điểm số MaxSoftmax cho dữ liệu hiệu chỉnh.')
        return self 
    def OOD_KernalSHAP(self, pred_func_for_shap, num_actual_superpixels, num_samples=1000):
        self.xAI_method = 'KernalSHAP'
        explainer = KernelExplainer(pred_func_for_shap, np.zeros((1, num_actual_superpixels)))
        self.shap_values = explainer.shap_values(np.ones((1, num_actual_superpixels)), nsamples=num_samples)
    def plot(self, original_image, class_names, segmentation):
        """
        Hàm vẽ biểu đồ SHAP tùy chỉnh.
        Lấy tất cả dữ liệu cần thiết từ `self`.
        """
        if self.xAI_method == 'DeepSHAP':
            self.visualization.plot_deepshap(original_image, class_names, self.shap_values, self.sample_scores, self.probs, self.ood_percentile, self.detector)
        elif self.xAI_method == 'KernalSHAP':
            self.visualization.plot_kernelshap(original_image, class_names, segmentation, self.shap_values)
        else:
            raise RuntimeError("Please run the method before plotting")

        
    #Backup
    # def MSP_DeepSHAP(self, model, backgroundata = None,  device = None):
    #     detector = self.method(model)
    #     # Create a DeepSHAP explainer using the detector
    #     explainer = DeepExplainer(model, backgroundata)
    #     shap_values = explainer.shap_values(self.sample) # Chuyển đổi kích thước tensor về (1,3,128,128) để phù hợp với đầu vào của model
    #     scores = detector(self.sample)
    #     self.ind_scores_for_calibration = detector(backgroundata).cpu().numpy()
    #     print("Hiệu chỉnh hoàn tất.")
    #     processed_shap_values_for_plot = []
    #     num_output_classes = shap_values.shape[-1] # Lấy số lớp từ chiều cuối cùng (2 trong trường hợp này)
    #     for i in range(num_output_classes):
    #         # Lấy SHAP values cho lớp thứ i. Loại bỏ chiều batch (0) và chiều cuối cùng (i)
    #         # Kết quả sẽ có shape (3, 144, 144)
    #         shap_for_current_class = shap_values[0, :, :, :, i]
    #         processed_shap_values_for_plot.append(shap_for_current_class)

    #     return processed_shap_values_for_plot
    # PLOT Area

