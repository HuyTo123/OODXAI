import numpy as np
from skimage.segmentation import slic
from .. import KernelExplainer
from tqdm.autonotebook import tqdm
import torch
# Giả sử OODExplainerBase đã được định nghĩa
from .OodXAIBase import OODExplainerBase
import torchvision.transforms as transforms
from PIL import Image
from .ood_segment import Ood_segment
import math
from scipy.stats import chi2

class OodKernelExplainer(OODExplainerBase):
    def __init__(self, model=None, Ood_name=None, background_data=None, sample=None, device=None, class_name=None,
                 n_segments = 50, compactness = 10, sigma = 1, start_label = 0, transform_mean=[0.485, 0.456, 0.406], transform_std=[0.229, 0.224, 0.225],
                 image_numpy_unnormalized = None, num_samples=100, logits = None):
        """
        Subclass for KernelSHAP. __init__ chỉ dùng để lưu cấu hình.
        """
        # --- Super class init ---
        # `sample` ở đây là ảnh đã xử lý, dùng để tính OOD score
        super().__init__(model, Ood_name, background_data, sample, device, class_name, logits)

        # --- User parameters for segmentation ---
        # Lưu lại tất cả các cấu hình
        self.n_segments = n_segments
        self.compactness = compactness
        self.sigma = sigma
        self.start_label = start_label
        self.image_numpy_unnormalized = image_numpy_unnormalized
        self.num_samples = num_samples
        self.transform_mean = transform_mean
        self.transform_std = transform_std
        self.compactness = compactness
        self.sigma = sigma
        self.start_label = start_label
        # Phiên bản chưa chuẩn hóa về 0 hoặc 1 nha
        self.background_color = image_numpy_unnormalized.mean((0,1))
        # --- State parameters ---
        # Khởi tạo các biến trạng thái, sẽ được điền giá trị sau
        self.segments_slic = None
        self.aggregated_intensities_tuple = ()
        self.uncertainty_segments= None
        self.image_numpy_0_1 = None 
        self.representative_segment_labels = None
        self.mean_features = None
        self.inv_cov_matrices = None

        print("-> OodKernelExplainer đã được tạo và cấu hình. Sẵn sàng hoạt động.")

    
    def extract_segment_features(self):
        """
        Tạo một đối tượng Ood_segment để phân tích và trích xuất các
        đặc trưng cường độ sáng từ toàn bộ background_data.
        """
        print("\n--- Bắt đầu quy trình trích xuất đặc trưng segment ---")
        
        # 3. Tạo một đối tượng Ood_segment mới và truyền đầy đủ
        #    các thông tin cần thiết từ chính đối tượng OodKernelExplainer hiện tại (self).
        segment_analyzer = Ood_segment(
            model=self.model,
            Ood_name=self.Ood_name,
            background_data=self.background_data,
            sample=self.sample, # Cung cấp một sample để lớp cha hoạt động
            device=self.device,
            class_name=self.class_name,
            n_segments=self.n_segments,
            compactness=self.compactness,
            sigma=self.sigma,
            start_label=self.start_label,
            transform_mean=self.transform_mean,
            transform_std=self.transform_std,
            num_samples=self.num_samples,
        )

        
        # 4. Gọi đúng phương thức `explain` và trả về kết quả
        self.aggregated_intensities_tuple, self.representative_segment_labels, self.mean_features, self.inv_cov_matrices = segment_analyzer.Extract()
        torch.cuda.empty_cache()

        
    def explain(self):
        """
        Đây là phương thức CÔNG KHAI DUY NHẤT để chạy toàn bộ quy trình.
        Nó sẽ tự động làm mọi thứ: phân vùng, tạo ảnh, dự đoán và tính SHAP.
        """
        print("\n--- Bắt đầu quy trình giải thích của KernelSHAP ---")
      

        # Calculate OOD score for the sample
        # 1. Phân vùng ảnh bằng Superpixel
        transform_for_slic = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        pil_image = Image.fromarray(self.image_numpy_unnormalized)
        image_tensor_resized = transform_for_slic(pil_image)
        self.image_numpy_0_1 = image_tensor_resized.permute(1, 2, 0).numpy()
        background_color = self.image_numpy_0_1.mean((0,1))
        self.segments_slic = slic(self.image_numpy_0_1, n_segments=self.n_segments,
                                  compactness=self.compactness, sigma=self.sigma, start_label=self.start_label)
        num_actual_superpixels = len(np.unique(self.segments_slic))
        print(f"1. Phân vùng ảnh thành {num_actual_superpixels} siêu pixel.")

        # 2. Định nghĩa hàm dự đoán nội bộ
        # Hàm này sẽ được truyền vào KernelExplainer và được gọi tự động
        def transform_masked_image(numpy_img):
            """
            Hàm này nhận một ảnh NumPy float [0,1] và chuẩn hóa nó đúng cách.
            """
            # Chuyển NumPy (H, W, C) sang Tensor (C, H, W) mà KHÔNG chia lại cho 255
            tensor_img = torch.from_numpy(numpy_img.transpose(2, 0, 1)).float()
            
            # Chuẩn hóa như bình thường
            return transforms.functional.normalize(tensor_img, self.transform_mean, self.transform_std)

        def prediction_function(z):
            batch_size = 10 # Xử lý 10 ảnh mỗi lần, bạn có thể điều chỉnh số này
            
            # Danh sách để lưu kết quả dự đoán của từng lô nhỏ
            all_logits = []
            unique_labels = np.unique(self.segments_slic)
            # Vòng lặp để xử lý z theo từng lô
            for i in tqdm(range(0, z.shape[0], batch_size), desc="SHAP Batches"):
                # Lấy ra một lô nhỏ các mặt nạ
                z_batch = z[i:i + batch_size]

                
                # --- Phần code bên trong giữ nguyên, nhưng chỉ cho lô nhỏ ---
                masked_images_np = []
                for mask in z_batch:
                    temp_image = self.image_numpy_0_1.copy()
                    inactive_segments = np.where(mask == 0)[0]
                    inactive_labels = unique_labels[inactive_segments]
                    mask_all_inactive = np.isin(self.segments_slic, inactive_labels)
                    temp_image[mask_all_inactive] = background_color
                    masked_images_np.append(temp_image)

                tensors = torch.stack(
                    [transform_masked_image(img) for img in masked_images_np]
                ).to(self.device)
                        
                self.model.eval()
                with torch.no_grad():
                    # Dự đoán trên lô nhỏ
                    logits = self.model(tensors)
                
                # Thêm kết quả của lô này vào danh sách
                all_logits.append(logits)
            
            # Nối tất cả các kết quả từ các lô nhỏ lại thành một mảng lớn duy nhất
            all_logits_numpy = [l.cpu().numpy() for l in all_logits]
            # Concate để ra tonsor (batch_size, classes_numbs)
            return np.concatenate(all_logits_numpy, axis=0)

        # 3. Khởi tạo KernelExplainer và tính toán SHAP values
        print(f"2. Bắt đầu tính toán SHAP values với {self.num_samples} mẫu...")
        explainer = KernelExplainer(prediction_function, np.zeros((1, num_actual_superpixels)))
        self.shap_values = explainer.shap_values(np.ones((1, num_actual_superpixels)), nsamples=self.num_samples)
        shap_values_for_class1 = self.shap_values[0, :, 0]
        shap_values_for_class2 = self.shap_values[0, :, 1]
        print(np.sum(shap_values_for_class1) + explainer.expected_value[0], 'ket qua', self.logits)       
        print(np.sum(shap_values_for_class2) + explainer.expected_value[1], 'ket qua', self.logits)

        print("3. Tính toán SHAP values hoàn tất!")
        # Tính toán các segment bị uncertainity
        sample_features, sample_labels = self.uncertainty()
        self.uncertainty_segments = self.find_unsafe_segments(sample_features, sample_labels)


        return self # Trả về self để có thể gọi .plot() nối tiếp
    def uncertainty (self, top_k=3):
        temp_results = {class_idx: [] for class_idx in range(self.num_classes)}
        temp_labels = {}
        for class_idx in range(self.num_classes):
            # Lấy SHAP values của class hiện tại
            shap_values_for_class = self.shap_values[0, :, class_idx]       
            # 1. Lấy cả GIÁ TRỊ và VỊ TRÍ (label) của các segment có SHAP value > 0
            positive_shap_indices = np.where(shap_values_for_class > 0)[0]
            print(f'giá trị tích cực tới class "{class_idx}"',positive_shap_indices)
            if len(positive_shap_indices) == 0:
                # Nếu không có segment dương, ta vẫn cần xử lý để giữ shape
                # Tạo một hàng rỗng với đúng số cột
                num_cols = 1 + 3 * top_k
                empty_row = [0] + [np.nan] * (num_cols - 1)
                temp_results[class_idx] = empty_row
                continue

            # Lấy các giá trị SHAP tương ứng
            positive_shap_values = shap_values_for_class[positive_shap_indices]
            total_positive_shap = np.sum(positive_shap_values)

            
            # 2. Ghép cặp (SHAP value, label) lại với nhau
            shap_label_pairs = list(zip(positive_shap_values, positive_shap_indices))
            
            # 3. SẮP XẾP danh sách các cặp này dựa trên SHAP value (phần tử đầu tiên)
            #    Đây là thay đổi cốt lõi theo yêu cầu của bạn.
            sorted_by_shap = sorted(shap_label_pairs, key=lambda x: x[0], reverse=True)
            print('Số lượng' , len(sorted_by_shap))
            top_k_segments = sorted_by_shap[:top_k]

            top_labels = [label for shap, label in top_k_segments]
            temp_labels[class_idx] = top_labels
            # 4. Bây giờ, tạo danh sách (cường độ, label) cuối cùng DỰA TRÊN THỨ TỰ ĐÃ SẮP XẾP MỚI
            top_shap_probs = []
            top_pixel_counts = []
            top_intensities = []

            for shap_value, segment_label in top_k_segments:
                # Tính toán 3 loại đặc trưng
                prop = round(shap_value / total_positive_shap, 3) if total_positive_shap > 0 else 0
                top_shap_probs.append(prop)
                
                mask = (self.segments_slic == segment_label)
                pixels_in_segment = self.image_numpy_0_1[mask]
                
                pixel_count = pixels_in_segment.shape[0]
                top_pixel_counts.append(pixel_count)
                
                if pixels_in_segment.size > 0:
                    avg_intensity = np.mean(pixels_in_segment) * 255
                    top_intensities.append(avg_intensity)
                else:
                    top_intensities.append(np.nan)
            
            # 3. LÀM ĐẦY (PADDING) DỮ LIỆU ĐỂ LUÔN CÓ ĐỦ top_k PHẦN TỬ
            while len(top_shap_probs) < top_k:
                top_shap_probs.append(np.nan)
            while len(top_pixel_counts) < top_k:
                top_pixel_counts.append(np.nan)
            while len(top_intensities) < top_k:
                top_intensities.append(np.nan)    
            # 4. TẠO HÀNG DỮ LIỆU (ROW) CÓ CẤU TRÚC CHUẨN
            # Dùng index 0 làm placeholder cho ID của sample duy nhất này
            row_data = [0] + top_shap_probs + top_intensities + top_pixel_counts
            temp_results[class_idx] = row_data
        # 5. ĐỊNH DẠNG KẾT QUẢ CUỐI CÙNG THÀNH TUPLE( (class, matrix_2D), ... )
        result_list = []
        for class_idx, row_values in sorted(temp_results.items()):
            # Chuyển đổi hàng (list) thành ma trận 2D có 1 hàng
            matrix_2d_one_row = np.array([row_values]) 
            result_list.append((class_idx, matrix_2d_one_row))
        sample_uncertainty_tuple = tuple(result_list)
        label_list = []
        for class_idx, labels in sorted(temp_labels.items()):
            label_list.append((class_idx, labels))
        sample_labels_tuple = tuple(label_list)
        return sample_uncertainty_tuple, sample_labels_tuple      

    def find_unsafe_segments(self, sample_uncertainty_tuple, sample_labels_tuple, top_k=3, p_value=0.05):
        print("\n--- Bắt đầu tìm kiếm Unsafe Segments bằng Mahalanobis Distance ---")
        
        unsafe_segments = {f"class_{i}": [] for i in range(self.num_classes)}
        
        # Chuyển tuple labels thành dict để dễ truy cập
        sample_labels_dict = dict(sample_labels_tuple)

        for class_idx in range(self.num_classes):
            print(f"\n-> Phân tích cho Class {self.class_name[class_idx]} (index: {class_idx})")

            mean_vectors_for_class = self.mean_features[class_idx]
            inv_cov_matrices_for_class = self.inv_cov_matrices[class_idx]
            sample_segment_labels = sample_labels_dict.get(class_idx, [])

            if mean_vectors_for_class.size == 0 or inv_cov_matrices_for_class.size == 0 or not sample_segment_labels:
                print("Không có đủ dữ liệu để so sánh.")
                continue
                
            for k in range(len(sample_segment_labels)):
                mean_vec_k = mean_vectors_for_class[k]
                inv_cov_k = inv_cov_matrices_for_class[k]

                sample_feature_matrix = sample_uncertainty_tuple[class_idx][1]
                sample_feature_vector_k = np.array([
                    sample_feature_matrix[0, 1 + k],
                    sample_feature_matrix[0, 1 + top_k + k],
                    sample_feature_matrix[0, 1 + 2 * top_k + k]
                ])
                
                if np.isnan(sample_feature_vector_k).any():
                    continue

                diff = sample_feature_vector_k - mean_vec_k
                mahalanobis_dist_sq = diff.T @ inv_cov_k @ diff
                # degree of freedom phụ thuộc vào các loại đặc trưng mà ta muốn khai thác 
                degrees_of_freedom = 3
                threshold = chi2.ppf((1 - p_value), df=degrees_of_freedom)
                
                segment_label = sample_segment_labels[k]
                is_anomalous = mahalanobis_dist_sq > threshold
                
                print(f"Segment Label {segment_label} (bậc {k+1}): Distance^2 = {mahalanobis_dist_sq:.2f}, Threshold = {threshold:.2f} -> Bất thường: {is_anomalous}")

                if is_anomalous:
                    key_name = f"class_{class_idx}"
                    unsafe_segments[key_name].append(segment_label)

        final_result_list = []
        for key, values in sorted(unsafe_segments.items()):
            class_index = int(key.split('_')[1])
            final_result_list.append((class_index, values))

        return tuple(final_result_list) 
    def get_displayable_image_from_tensor(self, tensor_image):     
            # Chuyển tensor sang CPU, sang NumPy và đổi chiều từ (C, H, W) -> (H, W, C)
            np_image = tensor_image.cpu().numpy().transpose((1, 2, 0))
            
            # Giải chuẩn hóa: (ảnh * std) + mean
            # self.transform_std và self.transform_mean đã được định nghĩa trong __init__
            np_image = self.transform_std * np_image + self.transform_mean
            
            # Cắt các giá trị để đảm bảo nằm trong khoảng [0, 1] hợp lệ cho việc hiển thị
            return np.clip(np_image, 0, 1)
    def _create_representative_masks(self):
        """
        (Hàm nội bộ) Tạo ra một dictionary chứa các ảnh đại diện đã được che (masked).
        Key là class_idx, value là ảnh NumPy sẵn sàng để vẽ.
        """
        if not self.representative_segment_labels:
            return {}

        print("-> Đang chuẩn bị ảnh đại diện (masked)...")
        rep_labels_dict = dict(self.representative_segment_labels)
        masked_images_dict = {}
       
        for class_idx, rep_info in rep_labels_dict.items():
            if rep_info is None:
                continue
            
            # 1. Lấy thông tin
            rep_image_id = int(rep_info[0])
            top_labels = rep_info[1:]

            # 2. Lấy và chuẩn bị ảnh
            rep_image_tensor = self.background_data[rep_image_id]
            rep_image_numpy = self.get_displayable_image_from_tensor(rep_image_tensor)

            # 3. Phân đoạn ảnh đại diện
            rep_segmentation = slic(rep_image_numpy, n_segments=self.n_segments,
                                    compactness=self.compactness, sigma=self.sigma,
                                    start_label=self.start_label)

            # 4. Tạo mask
            mask_image = np.zeros_like(rep_image_numpy)
            boolean_mask = np.isin(rep_segmentation, top_labels)
            mask_image[boolean_mask] = rep_image_numpy[boolean_mask]
            
            # 5. Lưu vào dictionary
            masked_images_dict[class_idx] = mask_image
            
        return masked_images_dict

    def plot(self, class_names=None):
        sample_s, percentile = self.calculate_Ood_scores()
        representative_masked_images = self._create_representative_masks()

        self.visualization.plot_kernelshap_with_uncertainty(self.image_numpy_0_1, 
                                           class_names=self.class_name, 
                                           segmentation=self.segments_slic, 
                                           shap_values=self.shap_values,
                                            unsafe_segments_tuple=self.uncertainty_segments,
                                           ood_percentile=percentile,
                                           sample_scores=sample_s,
                                           probs=self.probs,
                                           detector=self.Detector,
                                           representative_masked_images=representative_masked_images)


 # unsafe_segments = {f"class_{i}": [] for i in range(self.num_classes)}

        # # Lặp qua từng class để tính toán và so sánh
        # for class_idx in range(self.num_classes):
        #     # 1. Lấy và xử lý dữ liệu nền cho class hiện tại
        #     background_list_original = self.aggregated_intensities_tuple[class_idx][1]
        #     background_list_cleaned = [item for item in background_list_original if not math.isnan(item)]
            
        #     # Bỏ qua nếu không có đủ dữ liệu nền để tính ngưỡng
        #     if len(background_list_cleaned) < 20: # Cần ít nhất 20 điểm dữ liệu
        #         print(f"Cảnh báo: Class {class_idx} không có đủ dữ liệu nền để xác định ngưỡng an toàn.")
        #         continue
                
        #     sorted_background_list = sorted(background_list_cleaned)

        #     # 2. Xác định ngưỡng an toàn cho class hiện tại
        #     list_len = len(sorted_background_list)
        #     lower_index = int(list_len * 0.1)
        #     upper_index = int(list_len * 0.9) - 1
            
        #     lower_bound = sorted_background_list[lower_index]
        #     upper_bound = sorted_background_list[upper_index]
            
        #     print(f"Ngưỡng an toàn cho Class {class_idx}: ({lower_bound:.2f}, {upper_bound:.2f})")

        #     # 3. So sánh các segment của sample với ngưỡng
        #     sample_pairs_for_class = sample_uncertainty_tuple[class_idx][1]
        #     for intensity, label in sample_pairs_for_class:
        #         if not (lower_bound < intensity < upper_bound):
        #             # Gán vào dictionary kết quả theo đúng class_idx
        #             key_name = f"class_{class_idx}"
        #             unsafe_segments[key_name].append(label)
        #             print(f"-> Class {class_idx}: Segment {label} (giá trị {intensity:.2f}) được xác định là KHÔNG an toàn.")
        
        # # Trả về kết quả cuối cùng
        # final_result_list = []
        # # Sắp xếp dictionary theo key để đảm bảo thứ tự class (0, 1, 2...)
        # for key, values in sorted(unsafe_segments.items()):
        #     # Tách chỉ số class (là số nguyên) từ key (là chuỗi 'class_0')
        #     class_index = int(key.split('_')[1])
        #     final_result_list.append((class_index, values))

        # # Trả về kết quả cuối cùng dưới dạng một tuple lớn
        # return tuple(final_result_list)