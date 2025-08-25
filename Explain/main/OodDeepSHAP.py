""" This method is combine from DeepSHAP and OOD methods."""
from .. import DeepExplainer  
from scipy.stats import percentileofscore
from .OodXAIBase import OODExplainerBase

class OODDeepExplainer(OODExplainerBase):
    def __init__(self, model=None, Ood_name=None, background_data=None, sample=None, device=None ):
        """
            Subclass for DeepSHAP
        """
        super().__init__(model, Ood_name,background_data, sample ,device)
        self.calculate_Ood_scores()
        self.explainer = DeepExplainer(self.model, self.background_data.to(self.device))
        print("-> OODDeepExplainer has been created.")



    def explain(self):
        # DeepSHAP
        shap_values_raw = self.explainer.shap_values(self.sample, check_additivity=False)  # Chuyển đổi kích thước tensor về (1,3,128,128) để phù hợp với đầu vào của model
        # Xử lý shap_values để có dạng list cho hàm plot
        num_classes = shap_values_raw.shape[-1]
        self.shap_values = [shap_values_raw[0, :, :, :, i] for i in range(num_classes)]
        print('Đã tính toán và xử lý xong SHAP values.')
        
        return self 
    def plot(self, original_image, class_names):
        self.visualization.plot_deepshap(original_image, class_names, self.shap_values, self.sample_scores, self.probs, self.ood_percentile, self.Detector)


    

