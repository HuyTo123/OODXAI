import numpy as np
import torch
from skimage.segmentation import slic
# Giả sử KernelExplainer được import từ thư viện SHAP gốc hoặc từ local
from oodxai.Explain import KernelExplainer 
from tqdm.autonotebook import tqdm
import torchvision.transforms as transforms
from PIL import Image
import math

from oodxai.Explain.main.OodXAIBase import OODExplainerBase

class Ood_segment(OODExplainerBase):
    def __init__(self, model=None, Ood_name=None, background_data=None, sample=None, device=None, class_name=None,
                 n_segments=50, compactness=10, sigma=1, start_label=0, transform_mean=(0.485, 0.456, 0.406), transform_std=(0.229, 0.224, 0.225),
                 num_samples=100):
        """
        Hàm khởi tạo "chuẩn OOP".
        """
        # --- 1. Gọi hàm khởi tạo của lớp cha ---
        super().__init__(model, Ood_name, background_data, sample, device, class_name)
        
        # --- 2. Khởi tạo các thuộc tính của riêng lớp Ood_segment ---
        self.n_segments = n_segments
        self.compactness = compactness
        self.sigma = sigma
        self.start_label = start_label
        self.num_samples = num_samples
        self.transform_mean = np.array(transform_mean)
        self.transform_std = np.array(transform_std)
        
        # Tự động xác định số lượng class từ model
        with torch.no_grad():
            output = self.model(self.sample.to(self.device))
            self.num_classes = output.shape[-1]

        self.aggregated_intensities_tuple = ()
        self.predicted_class_sample = None # Sẽ được tính sau

        print("-> Ood_segment đã được tạo và cấu hình đúng chuẩn.")

    def Extract(self, top_k=2):
        """
        Trích xuất đặc trưng (phiên bản đã tối ưu).
        """
        print(f"\n--- Bắt đầu quy trình trích xuất đặc trưng (Tối ưu) ---")
        
        temp_results = {class_idx: [] for class_idx in range(self.num_classes)}

        # 1. HÀM HELPER MỚI: Chỉ giải chuẩn hóa về numpy float [0, 1]
        def get_float_image_from_tensor(tensor_image):
            # Chuyển std và mean sang numpy array với kiểu float32 một cách tường minh
            std = np.array(self.transform_std, dtype=np.float32)
            mean = np.array(self.transform_mean, dtype=np.float32)

            np_image = tensor_image.cpu().numpy().transpose((1, 2, 0))
            
            # Phép tính giờ đây là float32 * float32 + float32 -> kết quả là float32
            np_image = std * np_image + mean
            return np.clip(np_image, 0, 1)

        print(f"Bắt đầu xử lý {len(self.background_data)} ảnh nền...")
        for image_tensor in tqdm(self.background_data, desc="Processing Images"):
            
            with torch.no_grad():
                logits = self.model(image_tensor.unsqueeze(0).to(self.device))
                predicted_class = torch.argmax(logits, dim=1).item()

            # 2. CHUẨN BỊ ẢNH VÀ PHÂN MẢNH
            # Chuyển tensor đã chuẩn hóa về ảnh numpy float [0, 1]
            image_numpy_float = get_float_image_from_tensor(image_tensor)
            
            # Slic chạy trên ảnh float [0, 1]
            segments_slic = slic(image_numpy_float, n_segments=self.n_segments,
                                compactness=self.compactness, sigma=self.sigma, start_label=self.start_label)
            num_actual_superpixels = len(np.unique(segments_slic))
            background_color = image_numpy_float.mean((0, 1))

            # Transform này giờ chỉ cần Normalize, vì đầu vào đã là tensor
            transform_for_prediction = transforms.Compose([
                    transforms.ToTensor(), # Chuyển numpy -> tensor
                    transforms.Resize((224, 224)), # Resize về đúng kích thước model
                    transforms.Normalize(self.transform_mean, self.transform_std)
                ])
            # 3. PREDICTION_FUNCTION ĐÃ ĐƯỢC "VECTOR HÓA"
            def prediction_function(z):
                batch_size = 10
                all_logits = []
                h, w, c = image_numpy_float.shape
                unique_labels = np.unique(segments_slic)
                for i in range(0, z.shape[0], batch_size):
                    z_batch = z[i:i + batch_size]
                    current_batch_size = z_batch.shape[0]

                    masked_images_np = []
        
                    for mask in z_batch:
                        temp_image = image_numpy_float.copy()
                        inactive_segments = np.where(mask == 0)[0]
                        inactive_labels = unique_labels[inactive_segments]
                        mask_all_inactive = np.isin(segments_slic, inactive_labels)
                        temp_image[mask_all_inactive] = background_color
                        
                        # THAY ĐỔI 2: Dùng append để thêm ảnh vào list
                        masked_images_np.append(temp_image)
                    
                    # THAY ĐỔI 3: Dùng list comprehension và torch.stack
                    tensors = torch.stack(
                        [transform_for_prediction(img) for img in masked_images_np]
                    ).to(self.device)

                    with torch.no_grad():
                        logits_shap = self.model(tensors)
                    all_logits.append(logits_shap.cpu().numpy())
                return np.concatenate(all_logits, axis=0)
            
            # Phần SHAP giữ nguyên
            explainer = KernelExplainer(prediction_function, np.zeros((1, num_actual_superpixels)))
            shap_values_for_image = explainer.shap_values(np.ones((1, num_actual_superpixels)), nsamples=self.num_samples)

            # Phần trích xuất đặc trưng giữ nguyên...
            shap_values_for_predicted_class = shap_values_for_image[0, :, predicted_class]
            print(np.sum(shap_values_for_predicted_class) + explainer.expected_value[predicted_class], 'ket qua', logits)

            positive_shap_indices = np.where(shap_values_for_predicted_class > 0)[0]
            
            if len(positive_shap_indices) == 0:
                continue
            
            # Chọn ra những segment dương
            positive_shap_values = shap_values_for_predicted_class[positive_shap_indices]

            # Chọn ra các cặp bao gồm label và giá trị shapely_dương
            shap_label_pairs = list(zip(positive_shap_values, positive_shap_indices))

            # Thực hiện việc sort dựa vào giá trị x[0] là shapely_values
            sorted_by_shap = sorted(shap_label_pairs, key=lambda x: x[0], reverse=True)
            # Chọn ra từ 0 -> k
            top_k_segments = sorted_by_shap[:top_k]

            top_intensities = []
            for _, segment_label in top_k_segments:
                mask = (segments_slic == segment_label)
                pixels_in_segment = image_numpy_float[mask] # Tính trên ảnh float [0,1]
                if pixels_in_segment.size > 0:
                    avg_intensity = np.mean(pixels_in_segment) * 255 # Nhân 255 để về thang 0-255
                    top_intensities.append(avg_intensity)

            temp_results[predicted_class].extend(top_intensities)

        result_list = []
        for class_idx, values in sorted(temp_results.items()):
            result_list.append((class_idx, values))
        
        self.aggregated_intensities_tuple = tuple(result_list)
        print(f"\n--- Hoàn tất trích xuất đặc trưng. ---")
        return self.aggregated_intensities_tuple



 # def Extract(self, top_k=2):
    #     """
    #     Trích xuất đặc trưng bằng cách:
    #     1. Dự đoán lớp của mỗi ảnh nền.
    #     2. Chỉ phân tích SHAP values của lớp được dự đoán đó.
    #     3. Tổng hợp các đặc trưng theo từng lớp đã được dự đoán.
    #     """
    #     print(f"\n--- Bắt đầu quy trình trích xuất đặc trưng theo dự đoán của model ---")
        
    #     # 1. Khởi tạo bộ chứa kết quả
    #     temp_results = {class_idx: [] for class_idx in range(self.num_classes)}

    #     def denormalize_image(tensor_image):
    #         np_image = tensor_image.cpu().numpy().transpose((1, 2, 0))
    #         np_image = self.transform_std * np_image + self.transform_mean
    #         np_image = np.clip(np_image, 0, 1)
    #         return (np_image * 255).astype(np.uint8)

    #     # 2. Bắt đầu Vòng lặp chính qua từng ảnh nền
    #     print(f"Bắt đầu xử lý {len(self.background_data)} ảnh nền...")
    #     for image_tensor in tqdm(self.background_data, desc="Processing Images"):
            
    #         # --- 3. DỰ ĐOÁN LỚP CỦA ẢNH HIỆN TẠI ---
    #         with torch.no_grad():
    #             # Thêm một chiều batch (unsqueeze) để đưa vào model
    #             logits = self.model(image_tensor.unsqueeze(0).to(self.device))
    #             predicted_class = torch.argmax(logits, dim=1).item()
    #             print(predicted_class)

    #         # --- 4. PHÂN TÍCH SHAP CÓ MỤC TIÊU ---
    #         image_numpy_unnormalized = denormalize_image(image_tensor)
    #         segments_slic = slic(image_numpy_unnormalized, n_segments=self.n_segments,
    #                              compactness=self.compactness, sigma=self.sigma, start_label=self.start_label)
    #         num_actual_superpixels = len(np.unique(segments_slic))
    #         background_color = image_numpy_unnormalized.mean((0,1))
    #         # Hàm prediction_function không thay đổi
    #         transform_for_prediction = transforms.Compose([
    #             transforms.ToTensor(),
    #             transforms.Normalize(self.transform_mean, self.transform_std)
    #         ])
    #         def prediction_function(z):
    #             masked_images_np = []
    #             for mask in z:
    #                 temp_image = image_numpy_unnormalized.copy()
    #                 inactive_segments = np.where(mask == 0)[0]
    #                 for seg_idx in inactive_segments:
    #                     temp_image[segments_slic == seg_idx] = background_color
    #                 masked_images_np.append(temp_image)
                
    #             tensors = torch.stack(
    #                 [transform_for_prediction(Image.fromarray(img.astype(np.uint8))) for img in masked_images_np]
    #             ).to(self.device)
                
    #             self.model.eval()
    #             with torch.no_grad():
    #                 logits_shap = self.model(tensors)
    #             return logits_shap.cpu().numpy()
    #         explainer = KernelExplainer(prediction_function, np.zeros((1, num_actual_superpixels)))
    #         shap_values_for_image = explainer.shap_values(np.ones((1, num_actual_superpixels)), nsamples=self.num_samples)

    #         # --- 5. TRÍCH XUẤT ĐẶC TRƯNG TỪ LỚP ĐÃ DỰ ĐOÁN ---
    #         # Lấy SHAP values của đúng lớp đã được dự đoán
    #         shap_values_for_predicted_class = shap_values_for_image[0, :, predicted_class]
    #         print(np.sum(shap_values_for_predicted_class) + explainer.expected_value[predicted_class], 'ket qua', logits)

    #         # Chỉ xét các segment có SHAP value > 0 -> Kết quả là vị trí (index) của các segment này -> vị trí vẫn giữ nguyên so với shap_values_for_class 
    #         # ví dụ [0,2, 5] có nghĩa là segment 0, segment 2, segment 5 có SHAP dương
    #         positive_shap_indices = np.where(shap_values_for_predicted_class > 0)[0]
            
    #         # Nếu không có segment nào có SHAP dương, ta bỏ qua ảnh này
    #         if len(positive_shap_indices) == 0:
    #             continue
    #         positive_shap_values = shap_values_for_predicted_class[positive_shap_indices]

    #         # 2. Ghép cặp (SHAP value, label) lại với nhau
    #         shap_label_pairs = list(zip(positive_shap_values, positive_shap_indices))

    #         # 3. SẮP XẾP danh sách các cặp này dựa trên SHAP value giảm dần
    #         sorted_by_shap = sorted(shap_label_pairs, key=lambda x: x[0], reverse=True)
    #         top_k_segments = sorted_by_shap[:top_k]

    #         # Tính toán và lưu giá trị RGB trung bình cho các segment dương
    #         top_intensities = []
    #         for _, segment_label in top_k_segments:
    #             mask = (segments_slic == segment_label)
    #             pixels_in_segment = image_numpy_unnormalized[mask]
    #             if pixels_in_segment.size > 0:
    #                 avg_intensity = np.mean(pixels_in_segment)
    #                 top_intensities.append(avg_intensity)

    #         temp_results[predicted_class].extend(top_intensities)

    #     # --- 7. KẾT THÚC: Chuyển đổi sang định dạng tuple cuối cùng ---
    #     result_list = []
    #     for class_idx, values in sorted(temp_results.items()):
    #         result_list.append((class_idx, values))
        
    #     self.aggregated_intensities_tuple = tuple(result_list)
    #     print(f"\n--- Hoàn tất trích xuất đặc trưng. ---")
    #     return self.aggregated_intensities_tuple