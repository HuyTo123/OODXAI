import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from skimage.segmentation import mark_boundaries 
from scipy.ndimage import center_of_mass
from matplotlib.gridspec import GridSpec 
from pathlib import Path
from PIL import Image
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
            out[segmentation == i ] = values[i]
        return out

    def plot_kernelshap_with_uncertainty(self, original_image, class_names, segmentation, 
                                         shap_values, unsafe_segments_tuple=None, 
                                         sample_scores=1, probs=1, ood_percentile=1, detector='Unknown', representative_masked_images=None, top_k_labels=None):
        """
        Hàm vẽ biểu đồ SHAP tùy chỉnh cho KernelSHAP, với khả năng tô đỏ
        các segment được xác định là không an toàn (unsafe/uncertain).

        Parameters
        ----------
        unsafe_segments_tuple : tuple, optional
            Kết quả trả về từ hàm `uncertainty`. 
            Có dạng ((class_idx, [labels]), ...), by default None
        """
        if shap_values is None:
            raise RuntimeError("Vui lòng cung cấp `shap_values`.")

        print("Bắt đầu vẽ biểu đồ giải thích KernelSHAP với vùng không chắc chắn...")
        
        # --- PHẦN 1: Xử lý dữ liệu đầu vào ---
        predicted_class_index = np.argmax(probs)
        num_classes = len(class_names)
        sv = shap_values[0] # Giả định shap_values đã được xử lý phù hợp
        
        all_shap_values_abs = np.abs(np.array(sv))
        max_abs_val = np.percentile(all_shap_values_abs, 99.9)
        if max_abs_val == 0: max_abs_val = 1e-6
        
        # Chuyển tuple kết quả `uncertainty` thành dictionary để dễ tra cứu
        unsafe_segments_dict = {}
        if unsafe_segments_tuple:
            for class_idx, labels in unsafe_segments_tuple:
                unsafe_segments_dict[class_idx] = labels
        

        # --- PHẦN 2: Vẽ biểu đồ ---
        fig = plt.figure(figsize=(5 * (1 + num_classes), 8)) # Tăng chiều cao
        gs = GridSpec(2, 1 + num_classes, figure=fig)
        
        # Subplot cho ảnh gốc, chiếm 2 hàng của cột đầu tiên
        ax_original = fig.add_subplot(gs[:, 0])

        custom_colormap = self.create_custom_colormap()
        image_with_boundaries = mark_boundaries(original_image, segmentation, color=(1, 1, 0)) 
        print(len(np.unique(segmentation)))
        # Vẽ ảnh gốc lên subplot đã tạo
        ax_original.imshow(image_with_boundaries)
        ax_original.set_title("Original Image", fontsize=12)
        ax_original.axis("off")

        unique_labels = np.unique(segmentation)

        # Lặp qua từng label để tìm tâm và vẽ số
        for label in unique_labels:
            y, x = center_of_mass(segmentation, segmentation, label)
            # Dùng ax.text để vẽ số (label) lên vị trí tâm (x, y)
            ax_original.text(x, y, str(label), 
                            fontsize=14, 
                            color='Red', 
                            ha='center', # Căn giữa theo chiều ngang
                            va='center')

        im = None
        for i in range(num_classes):
            # Tạo subplot cho từng ô trong grid
            ax_shap = fig.add_subplot(gs[0, i + 1]) # Ô phía trên cho SHAP
            ax_rep = fig.add_subplot(gs[1, i + 1])  # Ô phía dưới cho ảnh đại diện

            # --- Vẽ Heatmap SHAP (Ô phía trên) ---
            # 1. Vẽ heatmap SHAP như cũ (trên ax_shap)
            heatmap_overlay = self.fill_segmentation(sv[:,i], np.array(segmentation))
            ax_shap.imshow(original_image, alpha=0.3)
            im = ax_shap.imshow(heatmap_overlay, cmap=custom_colormap, vmin=-max_abs_val, vmax=max_abs_val, interpolation='nearest')

            # 2. Tô màu xanh các vùng không chắc chắn (trên ax_shap)
            if i in unsafe_segments_dict and unsafe_segments_dict[i]:
                unsafe_labels = unsafe_segments_dict[i]
                # (Phần code này giữ nguyên, chỉ đổi biến `ax` thành `ax_shap`)
                print(f"-> Class {i}: Tô màu xanh cho các segment không an toàn: {unsafe_labels}")
                green_overlay = np.zeros((segmentation.shape[0], segmentation.shape[1], 4), dtype=np.uint8)
                for label in unsafe_labels:
                    mask = (segmentation == label)
                    green_overlay[mask] = [0, 255, 0, 128] 
                ax_shap.imshow(green_overlay)

            # 3. Đặt tiêu đề và viền cho ảnh (trên ax_shap)
            title = f"Class: '{class_names[i]}'\nProb: {probs[i]:.2%}"
            ax_shap.set_title(title, fontsize=10)
            if i == predicted_class_index:
                for spine in ax_shap.spines.values():
                    spine.set_edgecolor('#FF0000'); spine.set_linewidth(3)
            ax_shap.axis("off")
            
            rep_masked_images_dict = {}
            current_dir = Path(__file__).resolve().parent
            groundth_true_path = current_dir / "represent"
           
            represnet_path = groundth_true_path / f"{class_names[i]}" /  f"{class_names[i]}.png"
            if represnet_path.exists():
                rep_image = Image.open(represnet_path).convert("RGB")
                rep_image.resize((224,224))
                rep_masked_images_dict[i] = rep_image
                ax_rep.imshow(rep_image)
                ax_rep.set_title(f"Represent. for '{class_names[i]}'", fontsize=10)
            else:
                ax_rep.set_title(f"No Represent. for '{class_names[i]}'", fontsize=10)
            ax_rep.axis('off')
            # for i  in range(num_classes):
            #     represent_path = groundth_true_path / f"{class_names[i]}.png"
            #     if not represent_path.exists():
            #         print(f"LỖI: Không tìm thấy ảnh đại diện cho class '{class_names[i]}' tại: {represent_path}")
            #         continue
            #     rep_image = Image.open(represent_path).convert("RGB")
            #     rep_masked_images_dict[i] = rep_image
            # # --- Vẽ ảnh đại diện (Ô phía dưới) ---
            # if i in rep_masked_images_dict:
            #     mask_image = rep_masked_images_dict[i]
            #     ax_rep.imshow(mask_image)
            #     ax_rep.set_title(f"Rep. for '{class_names[i]}'", fontsize=10)
            # else:
            #     ax_rep.set_title(f"No Rep. for '{class_names[i]}'", fontsize=10)
            # ax_rep.axis("off")

        # --- PHẦN 4: Hoàn thiện biểu đồ ---
        predicted_class_name = class_names[predicted_class_index]
        ood_info = f"OOD Score ({detector.__class__.__name__}): {sample_scores:.2f} (Anomalous: {ood_percentile:.1f}%)"
        final_title = f"Predicted: '{predicted_class_name}' ({probs[predicted_class_index]:.1%}) | {ood_info}"
        fig.suptitle(final_title, fontsize=14)
        if top_k_labels:
            # 1. Xây dựng chuỗi ký tự từ tuple đầu vào
            top_k_strings = []
            for class_idx, labels in top_k_labels:
                # Chuyển list các label (số nguyên) thành chuỗi ký tự
                labels_str = ', '.join(map(str, labels))
                # Tạo chuỗi cho từng class
                s = f"Class '{class_names[class_idx]}': [{labels_str}]"
                top_k_strings.append(s)
            
            # 2. Nối các chuỗi của từng class lại với nhau
            full_top_k_str = "Top Segments: " + " | ".join(top_k_strings)
            
            # 3. Dùng fig.text() để thêm văn bản vào Figure
            #    - x=0.5, ha='center': Căn giữa theo chiều ngang
            #    - y=0.93: Đặt ở vị trí ngay dưới suptitle (có thể cần tinh chỉnh)
            fig.text(0.5, 0.93, full_top_k_str, ha='center', va='center', fontsize=10, color='darkblue', wrap=True)
        # ==============================================================================
        # <<< KẾT THÚC CODE MỚI >>>

        fig.tight_layout(rect=[0, 0.08, 1, 0.92]) # Điều chỉnh rect để có không gian cho colorbar
        
        # <<< THAY ĐỔI 5: Cập nhật lại cách tính vị trí colorbar >>>
        if num_classes > 0 and im is not None:
            # Lấy vị trí của các subplot ở hàng dưới cùng để căn chỉnh colorbar
            ax_rep_first = fig.get_axes()[2] # ax_original là 0, ax_shap_0 là 1, ax_rep_0 là 2
            ax_rep_last = fig.get_axes()[-1]
            pos1 = ax_rep_first.get_position()
            pos_last = ax_rep_last.get_position()
            
            # Tạo subplot cho colorbar nằm ngang bên dưới tất cả
            cbar_ax = fig.add_axes([pos1.x0, pos1.y0 - 0.1, pos_last.x1 - pos1.x0, 0.05])
            cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
            cbar.set_label('SHAP Value Contribution', fontsize=10)
            
        plt.show()
    def default_plot(self, original_image, segmentation, shap_values_3d, class_names, logit):
        """
        Vẽ ảnh gốc và SHAP cho TẤT CẢ các class.
        Hỗ trợ input shap_values dạng (batch, n_segments, n_classes).
        """
        # shap_values_3d shape: (1, 50, 2)
        # num_classes sẽ lấy từ chiều cuối cùng của array hoặc len của class_names
        num_classes = len(class_names)
        # 1. Xử lý dữ liệu đầu vào để tìm max_val chung
        # Chúng ta cần lấy toàn bộ giá trị để tính scale màu chung
        all_vals = shap_values_3d.flatten()
        max_val = np.nanpercentile(np.abs(all_vals), 99.9)
        if max_val == 0: max_val = 1e-6

        cmap = self.create_custom_colormap()

        # 2. Tạo Grid
        fig, axes = plt.subplots(1, 1 + num_classes, figsize=(5 * (1 + num_classes), 6))
        if not isinstance(axes, np.ndarray): axes = [axes]
        ax_original = axes[0]
        unique_labels = np.unique(segmentation)
        for label in unique_labels:
            y, x = center_of_mass(segmentation, segmentation, label)
            # Dùng ax.text để vẽ số (label) lên vị trí tâm (x, y)
            ax_original.text(x, y, str(label), 
                            fontsize=12, 
                            color='Red', 
                            ha='center', # Căn giữa theo chiều ngang
                            va='center')
        # --- Cột 1: Ảnh gốc ---
        img_with_bounds = mark_boundaries(original_image, segmentation, color=(1, 1, 0), mode='thick')
        axes[0].imshow(img_with_bounds)
        axes[0].set_title("Original Image")
        axes[0].axis('off')

        # --- Các cột tiếp theo: SHAP cho từng Class ---
        im = None
        for i in range(num_classes):
            ax = axes[i + 1]
            cls_name = class_names[i]
            
            # --- SỬA ĐỔI QUAN TRỌNG TẠI ĐÂY ---
            # Giả sử shape là (batch, n_segments, n_classes) và batch = 1
            # Ta lấy: batch 0, tất cả segments, class thứ i
            shap_vec = shap_values_3d[0, :, i] 
            
            # Kiểm tra shape để chắc chắn
            # print(f"Shape vector cho class {cls_name}: {shap_vec.shape}") 

            heatmap = self.fill_segmentation(shap_vec, segmentation)

            # Vẽ
            ax.imshow(original_image, alpha=0.15)
            im = ax.imshow(heatmap, cmap=cmap, vmin=-max_val, vmax=max_val)
            prob_value = logit[0, i].item()
            ax.set_title(f"Class: {cls_name} prob={prob_value:.3f}")
            ax.axis('off')

        # --- Colorbar ---
        if im:
            fig.subplots_adjust(bottom=0.2)
            cbar_ax = fig.add_axes([0.15, 0.1, 0.7, 0.05])
            cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
            cbar.set_label('SHAP Value (Blue: Chống lại, Red: Ủng hộ)')
        
        plt.suptitle("SHAP Explanation for All Classes", fontsize=16)
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
        