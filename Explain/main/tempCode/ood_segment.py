import numpy as np
import torch
from skimage.segmentation import slic
from .. import KernelExplainer 
from tqdm import tqdm
import torchvision.transforms as transforms
from PIL import Image


from oodxai.Explain.main.OodXAIBase import OODExplainerBase
class Ood_segment(OODExplainerBase):
    def __init__(self, model=None, Ood_name=None, background_data=None, sample=None, device=None, class_name=None,
                 n_segments=50, compactness=10, sigma=1, start_label=1, transform_mean=(0.485, 0.456, 0.406), transform_std=(0.229, 0.224, 0.225),
                 num_samples=100, background_color=0):
        """
        Hàm khởi tạo "chuẩn OOP":
        1. Nhận đầy đủ các tham số cần thiết.
        2. Gọi hàm khởi tạo của lớp cha (super) với các tham số tương ứng.
        3. Khởi tạo các thuộc tính riêng của lớp này.
        """
        # --- 1. Gọi hàm khởi tạo của lớp cha ---
        super().__init__(model, Ood_name, background_data, sample, device, class_name)
        
        # --- 2. Khởi tạo các thuộc tính của riêng lớp Ood_segment ---
        self.n_segments = n_segments
        self.compactness = compactness
        self.sigma = sigma
        self.start_label = start_label
        self.num_samples = num_samples

        # Chuyển mean/std sang numpy array để dễ tính toán
        self.transform_mean = np.array(transform_mean)
        self.transform_std = np.array(transform_std)
        self.background_color = background_color
        
        # Tự động xác định số lượng class từ model
        with torch.no_grad():
            output = self.model(self.sample.to(self.device))
            self.num_classes = output.shape[-1]

        # Khởi tạo thuộc tính sẽ lưu kết quả cuối cùng theo đúng kiểu dữ liệu
        self.aggregated_intensities_tuple = ()

        print("-> Ood_segment đã được tạo và cấu hình đúng chuẩn.")
        with torch.no_grad():
            logits = self.model(self.sample.to(self.device))
            probabilities = torch.softmax(logits, dim=1)
            self.predicted_class_sample = torch.argmax(probabilities, dim=1)


    def Extract(self, top_k=3):
        print(f"\n--- Bắt đầu quy trình giải thích OOD cho từng vùng ---")
        
        temp_results = {class_idx: [] for class_idx in range(self.num_classes)}

        def denormalize_image(tensor_image):
            np_image = tensor_image.cpu().numpy().transpose((1, 2, 0))
            np_image = self.transform_std * np_image + self.transform_mean
            np_image = np.clip(np_image, 0, 1)
            return (np_image * 255).astype(np.uint8)
        print(f"Bắt đầu xử lý {len(self.background_data)} ảnh nền...")
        for i, image_tensor in enumerate(tqdm(self.background_data, desc="Processing Images")):
            
            # Chuyển tensor về ảnh numpy để phân vùng
            image_numpy_unnormalized = denormalize_image(image_tensor)

            # 1a. Phân vùng ảnh bằng Superpixel
            segments_slic = slic(image_numpy_unnormalized, n_segments=self.n_segments,
                                compactness=self.compactness, sigma=self.sigma, start_label=self.start_label)
            num_actual_superpixels = len(np.unique(segments_slic))

            # 1b. Định nghĩa hàm dự đoán nội bộ cho ảnh hiện tại
            transform_for_prediction = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.transform_mean, self.transform_std)
            ])

            def prediction_function(z):
                masked_images_np = []
                for mask in z:
                    temp_image = image_numpy_unnormalized.copy()
                    inactive_segments = np.where(mask == 0)[0]
                    for seg_idx in inactive_segments:
                        # Gán màu nền cho các vùng không hoạt động
                        temp_image[segments_slic == seg_idx] = self.background_color
                    masked_images_np.append(temp_image)
                
                tensors = torch.stack(
                    [transform_for_prediction(Image.fromarray(img.astype(np.uint8))) for img in masked_images_np]
                ).to(self.device)
                
                self.model.eval()
                with torch.no_grad():
                    logits = self.model(tensors)
                return logits.cpu().numpy()
            explainer = KernelExplainer(prediction_function, np.zeros((1, num_actual_superpixels)))
            shap_values_for_image = explainer.shap_values(np.ones((1, num_actual_superpixels)), nsamples=self.num_samples)
            for class_idx in range(self.num_classes):
                # Lấy SHAP values của class hiện tại (không dùng abs)
                shap_values_for_class = shap_values_for_image[0, :, class_idx]
                
                # Chỉ xét các segment có SHAP value > 0 -> Kết quả là vị trí (index) của các segment này -> vị trí vẫn giữ nguyên so với shap_values_for_class 
                # ví dụ [0,2, 5] có nghĩa là segment 0, segment 2, segment 5 có SHAP dương
                positive_shap_indices = np.where(shap_values_for_class > 0)[0]
                
                # Nếu không có segment nào có SHAP dương, ta bỏ qua class này
                if len(positive_shap_indices) == 0:
                    continue

                # Lấy ra giá trị SHAP của các segment dương
                positive_shaps = shap_values_for_class[positive_shap_indices]
                # Sắp xếp và lấy chỉ số tương đối trong mảng positive_shaps, chỉ số tương đối là số thứ tự trong positive_shaps
                # argsort trả về vị trí sắp xếp tăng dần
                sorted_relative_indices = np.argsort(positive_shaps)[-top_k:]
                # Lấy lại chỉ số gốc (label của segment)
                top_k_original_indices = positive_shap_indices[sorted_relative_indices].tolist()
                top_k_original_indices.reverse() # Đảo ngược để chỉ số quan trọng nhất đứng đầu
                
                # Tính toán và lưu giá trị RGB trung bình cho top K segments
                avg_intensity_values = []
                for segment_label in top_k_original_indices:
                    mask = (segments_slic == segment_label)
                    pixels_in_segment = image_numpy_unnormalized[mask]
                    avg_intensity = np.mean(pixels_in_segment)
                    avg_intensity_values.append(avg_intensity)
                # 2. Thêm kết quả của ảnh này vào dictionary tạm thời
                temp_results[class_idx].extend(avg_intensity_values)

        # Lưu thông tin của ảnh này
        result_list = []
        for class_idx, values in sorted(temp_results.items()):
            result_list.append((class_idx, values))
        aggregated_intensities_tuple = tuple(result_list)
        # -----------------------------------------------------------------------------
        print(f"\n--- Đã hoàn tất! Đã lưu lại giá trị RGB trung bình của {top_k} vùng quan trọng nhất. ---")
        return aggregated_intensities_tuple
        
        
