""" This method is combine from DeepSHAP and MSP methods."""
import numpy as np

from pytorch_ood.detector import MaxSoftmax, EnergyBased, ODIN, Mahalanobis, MCD
from matplotlib import colors
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from .Explain import DeepExplainer  
from scipy.stats import percentileofscore


class OODDeepExplainer():
    def __init__(self, model, method_name, backgroundata=None, sample=None, device=None):
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
        if method_name == 'MSP':
            self.detector = MaxSoftmax(self.model)
        elif method_name == 'Energy':
            self.detector = EnergyBased(self.model)
        else:
            raise ValueError(f"The method '{method_name}' is not supported.")
         # --- THỰC HIỆN HIỆU CHỈNH NGAY LẬP TỨC KHI KHỞI TẠO ---
        print("Bắt đầu hiệu chỉnh OOD detector trên dữ liệu được cung cấp...")
        # Sử dụng detector đã khởi tạo để tính score trên toàn bộ calibration_loader
        self.ind_scores_for_calibration = self.detector.predict(self.backgroundata).detach().numpy()
        print(f"Hiệu chỉnh hoàn tất trên {len(self.ind_scores_for_calibration)} mẫu.")
        with torch.no_grad():
            logits = self.model(self.sample)
            self.probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    def MSP_DeepSHAP(self):
        # DeepSHAP
        print('Bắt đầu tính toán Deep SHAP...')
        explainer = DeepExplainer(self.model, self.backgroundata)
        shap_values_raw = explainer.shap_values(self.sample)  
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
    def create_custom_colormap(self):
        """Tạo một dải màu tùy chỉnh từ Xanh -> Trong suốt -> Đỏ."""
        blue = np.array([0, 139, 251, 255]) / 255.0
        red = np.array([255, 0, 81, 255]) / 255.0
        transparent_white = np.array([255, 255, 255, 0]) / 255.0 

        return colors.LinearSegmentedColormap.from_list(
            "shap_red_blue_transparent",
            [(0., blue), (0.5, transparent_white), (1., red)],
            N=256
        ) 

    def _fill_segmentation(self, values, segmentation):
        """Tô màu các vùng superpixel bằng giá trị SHAP tương ứng."""
        out = np.zeros(segmentation.shape)
        for i in range(len(values)):
            out[segmentation == i + 1] = values[i]
        return out

    def plot(self, original_image, class_names):
        """
        Hàm vẽ biểu đồ SHAP tùy chỉnh.
        Lấy tất cả dữ liệu cần thiết từ `self`.
        """
        # Kiểm tra xem hàm explain() đã được chạy chưa
        if self.shap_values is None or self.sample_scores is None:
            raise RuntimeError("Vui lòng chạy hàm .explain() trước khi vẽ biểu đồ.")

        print("Bắt đầu vẽ biểu đồ giải thích...")

        # --- Chuẩn bị dữ liệu từ self ---
        predicted_class_index = np.argmax(self.probs)
        num_classes = len(class_names)
        max_abs_val = np.abs(np.array(self.shap_values)).max()
        if max_abs_val == 0: max_abs_val = 1e-6

        # --- Vẽ biểu đồ ---
        fig, axes = plt.subplots(nrows=1, ncols=1 + num_classes, figsize=(5 * (1 + num_classes), 5))
        custom_colormap = self.create_custom_colormap()

        axes[0].imshow(original_image)
        axes[0].set_title("Original Image", fontsize=12)
        axes[0].axis("off")

        im = None
        for i in range(num_classes):
            ax = axes[i + 1]
            heatmap_overlay = np.sum(self.shap_values[i], axis=0)
            
            ax.imshow(original_image, alpha=0.6)
            im = ax.imshow(heatmap_overlay, cmap=custom_colormap, vmin=-max_abs_val, vmax=max_abs_val, interpolation='nearest')

            title = f"Class: '{class_names[i]}'\nProb: {self.probs[i]:.2%}"
            ax.set_title(title, fontsize=10)
            
            if i == predicted_class_index:
                for spine in ax.spines.values():
                    spine.set_edgecolor('#FF0000'); spine.set_linewidth(3)
            ax.axis("off")

        # Tiêu đề và colorbar
        predicted_class_name = class_names[predicted_class_index]
        ood_info = f"OOD Score ({self.detector.__class__.__name__}): {self.sample_scores:.2f} (Anomalous: {self.ood_percentile:.1f}%)"
        final_title = f"Predicted: '{predicted_class_name}' ({self.probs[predicted_class_index]:.1%}) | {ood_info}"
        fig.suptitle(final_title, fontsize=14)
        
        fig.tight_layout(rect=[0, 0.1, 1, 0.9])
        pos1 = axes[1].get_position()
        pos_last = axes[-1].get_position()
        cax = fig.add_axes([pos1.x0, pos1.y0 - 0.1, pos_last.x1 - pos1.x0, 0.05])
        cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
        cbar.set_label('SHAP Value Contribution', fontsize=10)
        
        plt.show()