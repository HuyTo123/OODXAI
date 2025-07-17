import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors



class Plot():
    def __init__(self):
        pass
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
    def fill_segmentation(self, values, segmentation):
        """Tô màu các vùng superpixel bằng giá trị SHAP tương ứng."""
        out = np.zeros(segmentation.shape)
        for i in range(len(values)):
            out[segmentation == i + 1] = values[i]
        return out

    def plot_kernelshap(self, original_image, class_names, segmentation, shap_values, sample_scores=1, probs=1, ood_percentile=1, detector='Unknown'):
        """
        Hàm vẽ biểu đồ SHAP tùy chỉnh cho KernelSHAP (phiên bản superpixel).
        """
        # Kiểm tra xem hàm explain() đã được chạy chưa
        if shap_values is None or sample_scores is None:
            raise RuntimeError("Vui lòng chạy hàm .explain() trước khi vẽ biểu đồ.")

        print("Bắt đầu vẽ biểu đồ giải thích KernelSHAP...")
        # --- Chuẩn bị dữ liệu ---
        predicted_class_index = np.argmax(probs)
        num_classes = len(class_names)
        # Xử lý output của KernelExplainer (có thể là list hoặc một mảng duy nhất)
        # và xử lý trường hợp phân loại nhị phân
        sv = shap_values[0]
        # Tính max_abs_val từ shap_values đã được xử lý
        all_shap_values_abs = np.abs(np.array(sv))
        max_abs_val = np.percentile(all_shap_values_abs, 99.9)
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
            
            # *** THAY ĐỔI CỐT LÕI NẰM Ở ĐÂY ***
            # Thay vì `np.sum`, ta dùng `_fill_segmentation` để tạo heatmap
            heatmap_overlay = self.fill_segmentation(sv[:,i], np.array(segmentation))
            
            # Vẽ ảnh gốc mờ làm nền
            ax.imshow(original_image, alpha=0.3)
            
            # Vẽ heatmap lên trên
            im = ax.imshow(heatmap_overlay, cmap=custom_colormap, vmin=-max_abs_val, vmax=max_abs_val, interpolation='nearest')

            # Phần còn lại giữ nguyên logic
            # title = f"Class: '{class_names[i]}'\nProb: {probs[i]:.2%}"
            title = f"Class: '{class_names[i]}' "
            ax.set_title(title, fontsize=10)
            
            if i == predicted_class_index:
                for spine in ax.spines.values():
                    spine.set_edgecolor('#FF0000'); spine.set_linewidth(3)
            ax.axis("off")

        # Tiêu đề và colorbar (giữ nguyên logic)
        predicted_class_name = class_names[predicted_class_index]
        # ood_info = f"OOD Score ({detector.__class__.__name__}): {sample_scores:.2f} (Anomalous: {ood_percentile:.1f}%)"
        # final_title = f"Predicted: '{predicted_class_name}' ({probs[predicted_class_index]:.1%}) | {ood_info}"
        # fig.suptitle(final_title, fontsize=14)
        
        fig.tight_layout(rect=[0, 0.1, 1, 0.9])
        pos1 = axes[1].get_position()
        pos_last = axes[-1].get_position()
        cax = fig.add_axes([pos1.x0, pos1.y0 - 0.1, pos_last.x1 - pos1.x0, 0.05])
        cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
        cbar.set_label('SHAP Value Contribution', fontsize=10)
        plt.show()

    def plot_deepshap(self, original_image, class_names, shap_values, sample_scores, probs, ood_percentile, detector):
        """
        Hàm vẽ biểu đồ SHAP tùy chỉnh.
        Lấy tất cả dữ liệu cần thiết từ `self`.
        """
        # Kiểm tra xem hàm explain() đã được chạy chưa
        if shap_values is None or sample_scores is None:
            raise RuntimeError("Vui lòng chạy hàm .explain() trước khi vẽ biểu đồ.")

        print("Bắt đầu vẽ biểu đồ giải thích...")

        # --- Chuẩn bị dữ liệu từ self ---
        predicted_class_index = np.argmax(probs)
        num_classes = len(class_names)

        all_shap_values_abs = np.abs(np.array(shap_values))
        max_abs_val = np.percentile(all_shap_values_abs, 99.9)
        # max_abs_val = np.abs(np.array(self.shap_values)).max() - Old method, don't use 2 lines above
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
            heatmap_overlay = np.sum(shap_values[i], axis=0)
 
            ax.imshow(original_image, alpha=0.3)
            im = ax.imshow(heatmap_overlay, cmap=custom_colormap, vmin=-max_abs_val, vmax=max_abs_val, interpolation='nearest')

            title = f"Class: '{class_names[i]}'\nProb: {probs[i]:.2%}"
            ax.set_title(title, fontsize=10)
            
            if i == predicted_class_index:
                for spine in ax.spines.values():
                    spine.set_edgecolor('#FF0000'); spine.set_linewidth(3)
            ax.axis("off")

        # Tiêu đề và colorbar
        predicted_class_name = class_names[predicted_class_index]
        ood_info = f"OOD Score ({detector.__class__.__name__}): {sample_scores:.2f} (Anomalous: {ood_percentile:.1f}%)"
        final_title = f"Predicted: '{predicted_class_name}' ({probs[predicted_class_index]:.1%}) | {ood_info}"
        fig.suptitle(final_title, fontsize=14)
        
        fig.tight_layout(rect=[0, 0.1, 1, 0.9])
        pos1 = axes[1].get_position()
        pos_last = axes[-1].get_position()
        cax = fig.add_axes([pos1.x0, pos1.y0 - 0.1, pos_last.x1 - pos1.x0, 0.05])
        cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
        cbar.set_label('SHAP Value Contribution', fontsize=10)
        
        plt.show()
        