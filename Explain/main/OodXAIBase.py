import numpy as np
# from pytorch_ood.detector import MaxSoftmax, EnergyBased, ODIN, Mahalanobis, MCD
from matplotlib import colors
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from .. import DeepExplainer  
from .. import KernelExplainer
from scipy.stats import percentileofscore
from scipy.ndimage import gaussian_filter

from .plot import Plot

class OODExplainerBase():
    def __init__(self, model=None, Ood_name=None, background_data=None, sample=None, device=None):
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
            sample : Sample for DeeepSHAP need to be Preprocessed, e.g., normalized, resized, etc.
                    Sample for KernelSHAP do not need to be Preprocessed, e.g., normalized, resized, etc.
            device : str or torch.device, optional
                Device to run the model on (e.g., 'cpu' or 'cuda').
            in_scores_for_calibration: 
                Use for calibration scores
        """
        params = [model, Ood_name, background_data, sample, device]
        if any(p is None for p in params):
            raise ValueError("All of parameters such as: model, Ood_name, background_data, sample, device need to be provided.")
        # User supplied parameters
        self.model = model
        self.background_data = background_data # Dữ liệu nền (background data) cho SHAP
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.Ood_name = Ood_name
        self.sample = sample

        self.model.to(self.device)
        self.sample.to(self.device)
        
        #  Package internal parameters
        self.ind_scores_for_calibration = None
        self.Detector = None
        self.sample_scores = None
        self.ood_percentile = None
        self.probs = None
        self.shap_values = None

        self.visualization = Plot()
        with torch.no_grad():
            logits = self.model(sample.to(self.device))
            self.probs = F.softmax(logits, dim=1).cpu().numpy()[0]


        print(f"Using device: {self.device}, Base has completely loaded.")
    
    def explain(self, input_data, ood_method='msp'):
        raise NotImplementedError("This method should be implemented in subclasses.")
    def plot(self, shap_values, input_data, ood_scores=None, class_names=None):
        raise NotImplementedError("This method should be implemented in subclasses.")
    def calculate_Ood_scores(self):
        """
        Một phương thức chung (có thể dùng lại) để tính các điểm OOD.
        """
        self.model.eval()
        with torch.no_grad():
            if self.Ood_name == 'MSP':
                # Max Softmax Probability Detection
                from pytorch_ood.detector import MaxSoftmax
                self.Detector = MaxSoftmax(self.model)
            elif self.Ood_name == 'ENERGY':
                # Energy-Based Detection
                from pytorch_ood.detector import EnergyBased
                self.Detector = EnergyBased(self.model)
            else:
                raise ValueError(f"The method '{self.Ood_name}' is not supported.")
            self.ind_scores_for_calibration = self.Detector.predict(self.background_data.to(self.device)).cpu().detach().numpy()
            self.sample_scores = self.Detector(self.sample.to(self.device)).item()
            self.ood_percentile = percentileofscore(self.ind_scores_for_calibration, self.sample_scores)
            print(f"Đã hoàn thành tính toán OOD với phương pháp :{self.Ood_name}")

           



    

    