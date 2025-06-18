""" This method is combine from DeepSHAP and MSP methods."""
import numpy as np

from pytorch_ood.detector import MaxSoftmax
from matplotlib import colors
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from .Explain import DeepExplainer  # Giả sử bạn đã cài đặt SHAP và có thể import từ oodxai.Explain.shap
# HOẶC, nếu bạn đã expose DeepExplainer trong oodxai/Explain/__init__.py như bạn đã làm:
# from .Explain import DeepExplainer


class OODDeepExplainer():
    def __init__(self, model, method, backgroundata=None, sample=None, device=None):
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
        """
        self.model = model
        self.backgroundata = backgroundata
        self.device = device
        self.sample = sample

        if method == 'MSP':
            self.method = MaxSoftmax

    def MSP_DeepSHAP(self, model, backgroundata = None,  device = None):
        detector = self.method(model)
        # Create a DeepSHAP explainer using the detector
        explainer = DeepExplainer(model, backgroundata)
        shap_values = explainer.shap_values(self.sample) # Chuyển đổi kích thước tensor về (1,3,128,128) để phù hợp với đầu vào của model
        
        
        scores = detector(self.sample)

        processed_shap_values_for_plot = []
        num_output_classes = shap_values.shape[-1] # Lấy số lớp từ chiều cuối cùng (2 trong trường hợp này)
        for i in range(num_output_classes):
            # Lấy SHAP values cho lớp thứ i. Loại bỏ chiều batch (0) và chiều cuối cùng (i)
            # Kết quả sẽ có shape (3, 144, 144)
            shap_for_current_class = shap_values[0, :, :, :, i]
            processed_shap_values_for_plot.append(shap_for_current_class)

        return processed_shap_values_for_plot, scores
    
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
    def plot_custom_shap(self, shap_values_list, original_image, class_names, scores):
        """
        Hàm vẽ biểu đồ SHAP tùy chỉnh cho Deep SHAP (phiên bản pixel-wise),
        với heatmap phủ trực tiếp lên ảnh gốc và colorbar được căn chỉnh.

        Args:
            shap_values_list (list of np.array): Danh sách các mảng SHAP values,
                                                 mỗi mảng có shape (C, H, W) hoặc (H, W, C),
                                                 tương ứng với đóng góp cho mỗi lớp.
                                                 Ví dụ: `shap_values_list[i]` là map đóng góp cho class `i`.
            original_image (np.array): Ảnh gốc, định dạng HWC (Height, Width, Channels), giá trị [0, 255].
            class_names (list): Danh sách tên các lớp.
        """
        print("Bắt đầu vẽ biểu đồ giải thích Deep SHAP (phiên bản pixel-wise)...")

        if original_image.dtype == np.float32 or original_image.dtype == np.float64:
            original_image_display = (original_image * 255).astype(np.uint8)
        else:
            original_image_display = original_image.astype(np.uint8)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.sample)
            print("\n" + "="*25)
            print("--- DEBUG BÊN TRONG HÀM PLOT ---")
            print(f"Logits tính toán cho biểu đồ: {logits.tolist()}")
            print("="*25 + "\n")

            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            predicted_class_index = np.argmax(probs)
            predicted_class_prob = probs[predicted_class_index]

        num_classes = len(class_names)
        fig, axes = plt.subplots(
            nrows=1, ncols=1 + num_classes,
            figsize=(5 * (1 + num_classes), 6),
            constrained_layout=True
        )

        custom_colormap = self.create_custom_colormap()

        # --- Vẽ ảnh gốc (cột 1) ---
        axes[0].imshow(original_image_display)
        axes[0].set_title("Original Image", fontsize=12)
        axes[0].axis("off")

        # --- Chuẩn hóa SHAP values và tính max_abs_val ---
        # Ở đây, shap_values_list được kỳ vọng chứa các mảng có shape (C, H, W)
        # hoặc (H, W, C) cho mỗi lớp.
        # Chúng ta cần chuẩn hóa chúng về (H, W, C) để tính toán max_abs_val một cách nhất quán
        
        normalized_shap_values_for_max_abs = []
        for idx, s_val in enumerate(shap_values_list):
            print(f"DEBUG: shap_values_list[{idx}] before processing shape (for max_abs_val): {s_val.shape}")
            if s_val.ndim == 3:
                # Nếu là (C, H, W) -> chuyển thành (H, W, C)
                if s_val.shape[0] == original_image.shape[-1] and s_val.shape[1:] == original_image.shape[:-1]:
                    normalized_shap_values_for_max_abs.append(np.transpose(s_val, (1, 2, 0)))
                # Nếu đã là (H, W, C)
                elif s_val.shape[-1] == original_image.shape[-1] and s_val.shape[:-1] == original_image.shape[:-1]:
                    normalized_shap_values_for_max_abs.append(s_val)
                else:
                    # Fallback nếu không khớp chính xác, cố gắng transpose nếu chiều đầu tiên nhỏ hơn chiều còn lại
                    if s_val.shape[0] < s_val.shape[1] and s_val.shape[0] <= original_image.shape[-1]:
                         normalized_shap_values_for_max_abs.append(np.transpose(s_val, (1, 2, 0)))
                    else:
                         print(f"WARNING: Unexpected 3D shap_values_list[{idx}] shape: {s_val.shape}. Adding as is.")
                         normalized_shap_values_for_max_abs.append(s_val) # Add as is, might cause errors if not (H,W,C)
            elif s_val.ndim == 2: # Already (H, W)
                normalized_shap_values_for_max_abs.append(s_val)
            else:
                raise ValueError(f"shap_values_list[{idx}] has unexpected dimensions: {s_val.shape}. Expected 2D or 3D for max_abs_val calculation.")

        all_shap_values_flat = np.concatenate([np.abs(s).flatten() for s in normalized_shap_values_for_max_abs])
        max_abs_val = np.max(all_shap_values_flat)
        if max_abs_val == 0:
            max_abs_val = 1e-6

        # --- Vẽ các biểu đồ giải thích cho từng lớp ---
        explanation_axes = []
        for i in range(num_classes):
            ax = axes[i + 1]
            explanation_axes.append(ax)

            current_shap_map = shap_values_list[i] # Lấy mảng SHAP cho lớp hiện tại
            print(f"DEBUG: current_shap_map for class {class_names[i]} shape before sum: {current_shap_map.shape}")
            
            # Xử lý tùy thuộc vào hình dạng của current_shap_map
            # Nếu current_shap_map là (C, H, W)
            if current_shap_map.ndim == 3 and current_shap_map.shape[0] == original_image.shape[-1]: # E.g., (3, 144, 144)
                # Sum over channels, result will be (H, W)
                heatmap_overlay = np.sum(current_shap_map, axis=0) # Sum along channel axis (axis=0)
            # Nếu current_shap_map là (H, W, C)
            elif current_shap_map.ndim == 3 and current_shap_map.shape[-1] == original_image.shape[-1]: # E.g., (144, 144, 3)
                # Sum over channels, result will be (H, W)
                heatmap_overlay = np.sum(current_shap_map, axis=-1)
            # Nếu current_shap_map đã là (H, W)
            elif current_shap_map.ndim == 2:
                heatmap_overlay = current_shap_map
            else:
                raise ValueError(f"Unexpected dimension or channel setup for SHAP map for class {class_names[i]}: {current_shap_map.shape}. Expected (C, H, W) or (H, W, C) or (H, W).")
            
            print(f"DEBUG: heatmap_overlay for class {class_names[i]} shape after sum: {heatmap_overlay.shape}")

            # Đảm bảo heatmap_overlay là 2D (H, W) trước khi imshow
            if heatmap_overlay.ndim != 2:
                raise ValueError(f"Final heatmap_overlay for class {class_names[i]} is not 2D. Shape: {heatmap_overlay.shape}")


            ax.imshow(original_image_display, alpha=0.6)
            im = ax.imshow(heatmap_overlay, cmap=custom_colormap,
                            vmin=-max_abs_val, vmax=max_abs_val,
                            interpolation='nearest')

            title = f"Class: '{class_names[i]}'\nProb: {probs[i]:.2%}"
            ax.set_title(title, fontsize=10)
            
            if i == predicted_class_index:
                for spine in ax.spines.values():
                    spine.set_edgecolor('#FF0000')
                    spine.set_linewidth(3)
            ax.axis("off")

        cbar = fig.colorbar(
            im,
            ax=explanation_axes,
            orientation="horizontal",
            aspect=40,
            pad=0.04,
            shrink=0.7
        )
        cbar.set_label('SHAP Value Contribution', fontsize=10)
        cbar.ax.tick_params(labelsize=8)

        fig.suptitle(f"Deep SHAP Explanation for Predicted Class: '{class_names[predicted_class_index]}' (Confidence: {scores})",
                     fontsize=14, y=0.98)

        plt.show()

        print("Hoàn thành vẽ biểu đồ giải thích Deep SHAP (phiên bản pixel-wise).")