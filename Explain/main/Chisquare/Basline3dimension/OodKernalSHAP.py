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
from pathlib import Path
import traceback
from collections import Counter
import warnings
warnings.filterwarnings("ignore")
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
        self.sample_labels_topk = None
        sample_s, ood_percentile = self.calculate_Ood_scores()
        self.sample_scores = sample_s
        self.ood_percentile = ood_percentile
        self.top_k = None
        print("-> OodKernelExplainer đã được tạo và cấu hình. Sẵn sàng hoạt động.")

    def extract_and_save_statistics(self):
        print("\n--- Bắt đầu quy trình trích xuất đặc trưng segment và lưu  ---")
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
        segment_analyzer.extract_and_save_statistics()

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
        self.aggregated_intensities_tuple, self.representative_segment_labels, self.mean_features, self.inv_cov_matrices, self.top_k = segment_analyzer.Extract()
        torch.cuda.empty_cache()

        
    def explain(self, n_shap_runs=10): # Thêm tham số n_shap_runs
        """
        Đây là phương thức CÔNG KHAI DUY NHẤT để chạy toàn bộ quy trình.
        (SỬA ĐỔI): Chạy SHAP n_shap_runs lần, tính SHAP value trung bình
        và tìm các segment xuất hiện nhiều nhất (từ TẤT CẢ các segment dương).
        """
        print("\n--- Bắt đầu quy trình giải thích của KernelSHAP ---")
      
        # 1. Phân vùng ảnh bằng Superpixel (Giữ nguyên)
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

        # 2. Định nghĩa hàm dự đoán nội bộ (Giữ nguyên)
        def transform_masked_image(numpy_img):
            tensor_img = torch.from_numpy(numpy_img.transpose(2, 0, 1)).float()
            return transforms.functional.normalize(tensor_img, self.transform_mean, self.transform_std)

        def prediction_function(z):
            batch_size = 10
            all_logits = []
            unique_labels = np.unique(self.segments_slic)
            # Thêm leave=False để tqdm không làm lộn xộn output của vòng lặp ngoài
            for i in tqdm(range(0, z.shape[0], batch_size), desc="SHAP Batches", leave=False):
                z_batch = z[i:i + batch_size]
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
                    logits = self.model(tensors)
                all_logits.append(logits)
            
            all_logits_numpy = [l.cpu().numpy() for l in all_logits]
            return np.concatenate(all_logits_numpy, axis=0)

        # 3. Khởi tạo KernelExplainer VÀ CHẠY n_shap_runs LẦN
        print(f"2. Bắt đầu tính toán SHAP values ({n_shap_runs} lần, {self.num_samples} mẫu/lần)...")
        explainer = KernelExplainer(prediction_function, np.zeros((1, num_actual_superpixels)))
        
        all_shap_values_list = [] # List để lưu kết quả các lần chạy
        frequent_segment_labels = {} # Dict để lưu 10 segment thường gặp nhất
        
        for class_idx in range(self.num_classes):
            frequent_segment_labels[class_idx] = []
            
        # Vòng lặp chạy SHAP
        for i in tqdm(range(n_shap_runs), desc="SHAP Runs"):
            run_shap_values = explainer.shap_values(np.ones((1, num_actual_superpixels)), nsamples=self.num_samples)
            all_shap_values_list.append(run_shap_values)
            
            # --- [LOGIC MỚI BẮT ĐẦU TỪ ĐÂY] ---
            # Thu thập TẤT CẢ các segment dương của LẦN CHẠY NÀY
            for class_idx in range(self.num_classes):
                shap_values_for_class = run_shap_values[0, :, class_idx]
                
                # 1. Lấy TẤT CẢ các segment có SHAP value dương
                positive_shap_indices = np.where(shap_values_for_class > 0)[0]
                
                if len(positive_shap_indices) == 0:
                    continue
                
                # 2. KHÔNG SẮP XẾP, KHÔNG LẤY TOP 10
                # Thêm TẤT CẢ segment dương của lần chạy này vào list tổng
                frequent_segment_labels[class_idx].extend(positive_shap_indices)
            # --- [HẾT LOGIC MỚI] ---

        # 4. Tính toán SHAP value TRUNG BÌNH và gán vào self.shap_values
        self.shap_values = np.mean(all_shap_values_list, axis=0)
        print("3. Tính toán SHAP values (Trung bình) hoàn tất!")

        # 5. Xử lý hậu kỳ: Lọc ra các segment xuất hiện nhiều nhất
        final_frequent_labels = {}
        for class_idx in range(self.num_classes):
            if not frequent_segment_labels[class_idx]:
                final_frequent_labels[class_idx] = []
                continue
                
            # Đếm tần suất TẤT CẢ segment dương đã thu thập
            segment_counts = Counter(frequent_segment_labels[class_idx])
            
            # Sắp xếp tất cả các segment theo tần suất giảm dần
            all_most_frequent_pairs = segment_counts.most_common() 
            all_most_frequent_labels = [label for label, count in all_most_frequent_pairs]
            
            final_frequent_labels[class_idx] = all_most_frequent_labels
            # In ra top 10 cho người dùng xem
            print(f"  Class {class_idx}: 10 segments xuất hiện nhiều nhất (trong {len(all_most_frequent_labels)}): {all_most_frequent_labels[:10]}")

        # 6. Tính toán các segment bị uncertainity
        # Truyền TOÀN BỘ danh sách segment (đã sắp xếp theo tần suất) vào hàm uncertainty
        sample_features, sample_labels = self.uncertainty(final_frequent_labels)
        self.sample_labels_topk = sample_labels
        self.uncertainty_segments = self.find_unsafe_segments(sample_features, sample_labels)

        return self # Trả về self để có thể gọi .plot() nối tiếp


        return self # Trả về self để có thể gọi .plot() nối tiếp
    def uncertainty (self, frequent_segment_labels):
        temp_results = {class_idx: [] for class_idx in range(self.num_classes)}
        temp_labels = {}
        for class_idx in range(self.num_classes):
            top_10_frequent_labels = frequent_segment_labels.get(class_idx, [])
            
            # 2. Trích xuất self.top_k segment từ danh sách đó
            top_k_labels = top_10_frequent_labels[:self.top_k]
            temp_labels[class_idx] = top_k_labels
            
            if not top_k_labels: # Nếu list rỗng
                num_cols = 1 + 3 * self.top_k
                empty_row = [0] + [np.nan] * (num_cols - 1)
                temp_results[class_idx] = empty_row
                continue

            # 3. Lấy SHAP value TRUNG BÌNH (từ self.shap_values) cho top_k_labels
            # self.shap_values giờ là SHAP value trung bình (đã tính trong explain)
            shap_values_for_class = self.shap_values[0, :, class_idx]
            
            top_k_segments = [] # List các cặp (shap_value, label)
            total_positive_shap = 0
            
            for label in top_k_labels:
                shap_val = shap_values_for_class[label]
                # Chỉ xem xét nếu SHAP value trung bình là dương
                if shap_val > 0:
                    top_k_segments.append((shap_val, label))
                    total_positive_shap += shap_val
            
            # Sắp xếp lại top_k_segments dựa trên SHAP value trung bình (để hiển thị)
            top_k_segments = sorted(top_k_segments, key=lambda x: x[0], reverse=True)
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
            while len(top_shap_probs) < self.top_k:
                top_shap_probs.append(np.nan)
            while len(top_pixel_counts) < self.top_k:
                top_pixel_counts.append(np.nan)
            while len(top_intensities) < self.top_k:
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

    def find_unsafe_segments(self, sample_uncertainty_tuple, sample_labels_tuple, p_value=0.05):
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
                    sample_feature_matrix[0, 1 + self.top_k + k],
                    sample_feature_matrix[0, 1 + 2 * self.top_k + k]
                ])
                
                if np.isnan(sample_feature_vector_k).any():
                    continue

                diff = sample_feature_vector_k - mean_vec_k
                mahalanobis_dist_sq = diff.T @ inv_cov_k @ diff
                # degree of freedom phụ thuộc vào các loại đặc trưng mà ta muốn khai thác 
                degrees_of_freedom = 3
                percentile = self.ood_percentile/100
                #optional: static thresholding
                # threshold = chi2.ppf((1 - percentile * p_value), df=degrees_of_freedom)
                #optional: dynamic thresholding
                if percentile >= 0.5:
                    print("percentile >= 0.5")
                    threshold = chi2.ppf((1- percentile*15* p_value), df=degrees_of_freedom)
                else:
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
        representative_masked_images = self._create_representative_masks()

        self.visualization.plot_kernelshap_with_uncertainty(self.image_numpy_0_1, 
                                           class_names=self.class_name, 
                                           segmentation=self.segments_slic, 
                                           shap_values=self.shap_values,
                                            unsafe_segments_tuple=self.uncertainty_segments,
                                           ood_percentile=self.ood_percentile,
                                           sample_scores=self.sample_scores,
                                           probs=self.probs,
                                           detector=self.Detector,
                                           representative_masked_images=representative_masked_images,
                                           top_k_labels= self.sample_labels_topk)
    @staticmethod
    def _load_test_image(img_idx, base_folder, mean, std):
        """
        (Helper) Tải một ảnh test và chuẩn bị 2 định dạng cần thiết.
        """
        try:
            img_path = base_folder / f"Cat_{img_idx}.png"
            pil_image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            try:
                img_path = base_folder / f"Cat_{img_idx}.jpg"
                pil_image = Image.open(img_path).convert('RGB')
            except FileNotFoundError:
                print(f"!!! LỖI: Không tìm thấy ảnh Cat_{img_idx}.png hoặc .jpg")
                return None, None
                
        image_numpy_unnormalized = np.array(pil_image)
        
        tensor_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        sample_tensor = tensor_transform(pil_image).unsqueeze(0)
        
        return sample_tensor, image_numpy_unnormalized

    @staticmethod
    def _parse_labels_from_line(line):
        """
        (Helper) Chuyển một dòng text (ví dụ: "1, 5, 10" or "NaN")
        thành một list các số nguyên.
        """
        if line == 'NaN':
            return [-1]
        if not line:
            return []
        
        processed_line = line.replace(',', ' ')
        labels = []
        parts = processed_line.split()
        
        try:
            for part in parts:
                if part:
                    labels.append(int(part))
            return labels
        except ValueError as e:
            print(f"Lỗi khi xử lý dòng: '{line}'. Lỗi: {e}")
            return []
    def _load_ground_truth_staticscal(self, relative_file_path='test/model_stats_MSP_3features.npz'):
        """
        (Helper) Đọc file Ground Truth (GT) .txt theo từng khối 3 dòng.
        """
        # __file__ trỏ đến file OodKernalSHAP.py này
        script_dir = Path(__file__).parent
        input_filename = script_dir / relative_file_path

        
        try:
        # np.load tự động xử lý file .npz
        # allow_pickle=True là cần thiết nếu k_final được lưu dưới dạng đối tượng Python
            data = np.load(input_filename, allow_pickle=True)
            
            # Trích xuất dữ liệu bằng key
            # .item() để chuyển numpy array 0-chiều về lại số Python
            k_final = data['k_final'].item() 
            mean_features_np = data['mean_features']
            inv_cov_matrices_np = data['inv_cov_matrices']
            
            data.close() # Đóng file
            
            print(f"Tải thành công:")
            print(f"k_final: {k_final}")
            print(f"mean_features shape: {mean_features_np.shape}")
            print(f"inv_cov_matrices shape: {inv_cov_matrices_np.shape}")
            
            self.top_k, self.mean_features, self.inv_cov_matrices =  k_final, mean_features_np, inv_cov_matrices_np

        except FileNotFoundError:
            print(f"LỖI: Không tìm thấy file: {input_filename}")
            return None, None, None
        except KeyError as e:
            print(f"LỖI: File .npz không chứa key cần thiết: {e}. File có thể bị hỏng.")
            return None, None, None
        except Exception as e:
            print(f"LỖI: Đã xảy ra lỗi khi đọc file .npz: {e}")
            traceback.print_exc()
            return None, None, None
        
    def _load_ground_truth_data(self, relative_file_path='test/test_list.txt'):
        """
        (SỬA LỖI) Đọc file Ground Truth (GT) .txt.
        GỘP DÒNG 2 và DÒNG 3 thành MỘT danh sách "unsafe" duy nhất.
        """
        script_dir = Path(__file__).parent
        file_path = script_dir / relative_file_path
        
        print(f"Bắt đầu đọc file Ground Truth từ: {file_path}")
        experiment_data = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                all_lines = f.readlines()

            for i in range(0, len(all_lines), 3):
                if i + 2 < len(all_lines):
                    image_index_line = all_lines[i].strip()
                    if not image_index_line:
                        continue
                    image_index = int(image_index_line)
                    
                    # Đọc cả 2 dòng
                    class_0_line = all_lines[i+1].strip()
                    class_1_line = all_lines[i+2].strip()
                    
                    c0_labels = self._parse_labels_from_line(class_0_line)
                    c1_labels = self._parse_labels_from_line(class_1_line)

                    # --- LOGIC GỘP NHÃN (ĐÃ SỬA) ---
                    # 1. Kiểm tra điều kiện "skip"
                    if c0_labels == [-1] and c1_labels == [-1]:
                        # Thêm tuple (index, list_marker)
                        experiment_data.append((image_index, [-1])) 
                    else:
                        # 2. Gộp 2 set lại và loại bỏ marker -1
                        merged_set = (set(c0_labels) | set(c1_labels)) - {-1}
                        # Thêm tuple (index, list_đã_gộp)
                        experiment_data.append((image_index, list(merged_set)))
            
            print(f"Đọc file GT hoàn tất. Tìm thấy {len(experiment_data)} mẫu.")
            return experiment_data
            
        except FileNotFoundError:
            print(f"LỖI: Không tìm thấy file GT tại: {file_path}")
            return []
        except Exception as e:
            print(f"Lỗi không xác định khi đọc GT: {e}")
            return []

    @staticmethod
    def _calculate_metrics(gt_labels, pred_labels, n_segments):
        """
        (Helper) Tính toán Precision, Recall, F1, Accuracy.
        """
        gt_positive_set = set(gt_labels) - {-1}
        pred_positive_set = set(pred_labels)
        all_segments_set = set(range(n_segments))
        gt_negative_set = all_segments_set - gt_positive_set
        pred_negative_set = all_segments_set - pred_positive_set
        
        tp = len(gt_positive_set.intersection(pred_positive_set))
        fp = len(pred_positive_set - gt_positive_set)
        fn = len(gt_positive_set - pred_positive_set)
        tn = len(gt_negative_set.intersection(pred_negative_set))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        
        return precision, recall, f1, accuracy

    @staticmethod
    def _calculate_hit_rate(gt_labels, top_k_labels):
        """
        (Helper) Tính "Hit Rate": Tỷ lệ các segment GT "unsafe" lọt vào Top-K của SHAP.
        """
        gt_positive_set = set(gt_labels) - {-1}
        top_k_set = set(top_k_labels)
        
        if len(gt_positive_set) == 0:
            return np.nan # Không có GT để "hit"
            
        hits = len(gt_positive_set.intersection(top_k_set))
        hit_rate = hits / len(gt_positive_set)
        
        return hit_rate
    def run_evaluation_experiment(self, num_runs=1):
        """
        (SỬA ĐỔI LẦN CUỐI) CHẠY THÍ NGHIỆM GIAI ĐOẠN 2 & 3.
        Chỉ chạy và báo cáo cho Class 0 (Mèo).
        So sánh kết quả giải thích của Class 0 với MỘT Ground Truth "unsafe" (đã gộp).
        """
        warnings.filterwarnings("ignore") # Tắt cảnh báo
        
        # --- Kiểm tra điều kiện ---
        if self.mean_features is None or self.inv_cov_matrices is None or self.top_k is None:
            print("LỖI: Bạn phải chạy `extract_segment_features()` TRƯỚC khi chạy hàm này.")
            return

        print("\n--- Giai đoạn 2: Bắt đầu Vòng lặp Đánh giá (Chỉ Class 0) ---")
        
        # 1. Tải Ground Truth (đã gộp)
        gt_data = self._load_ground_truth_data('test/test_list.txt')
        if not gt_data:
            print("LỖI: Không có dữ liệu Ground Truth. Dừng lại.")
            return
            
        # 2. Chuẩn bị list kết quả (CHỈ CÓ CLASS 0)
        all_results_c0 = {'precision': [], 'recall': [], 'f1': [], 'accuracy': [], 'hit_rate': []}
        test_img_folder = Path(__file__).parent / 'test' / 'CatandDog_segment_noise' / 'cat'

        # Lấy các giá trị đã "huấn luyện" từ self
        trained_mean = self.mean_features
        trained_inv_cov = self.inv_cov_matrices
        trained_top_k = self.top_k
        n_segments_trained = self.n_segments
        num_samples_shap_for_test = self.num_samples

        # 3. Lặp qua từng ảnh trong file GT (đã gộp)
        # gt_unsafe_labels giờ là list đã gộp, hoặc [-1] để skip
        for img_idx, gt_unsafe_labels in tqdm(gt_data, desc="Đánh giá các ảnh"):
            
            # 3.1. Kiểm tra điều kiện bỏ qua nếu có 2 loại -1
            if gt_unsafe_labels == [-1]:
                tqdm.write(f"Bỏ qua ảnh {img_idx} (cả 2 class là -1/NaN)")
                continue
                
            # 3.2. Tải ảnh test
            sample_tensor, numpy_img = self._load_test_image(
                img_idx, test_img_folder, self.transform_mean, self.transform_std
            )
            if sample_tensor is None:
                continue
                
            tqdm.write(f"\n--- Đang xử lý ảnh: Cat_{img_idx}.png ({num_runs} lần) ---")
            
            # 3.3. Chuẩn bị list (CHỈ CÓ CLASS 0)
            run_results_c0 = {'precision': [], 'recall': [], 'f1': [], 'accuracy': [], 'hit_rate': []}
            
            # 3.4. Vòng lặp 'num_runs' lần
            for _ in range(num_runs):
                explainer_test = OodKernelExplainer(
                    model=self.model, Ood_name=self.Ood_name,
                    background_data=self.background_data,
                    sample=sample_tensor.to(self.device), device=self.device,
                    class_name=self.class_name,
                    image_numpy_unnormalized=numpy_img,
                    num_samples=num_samples_shap_for_test,
                    n_segments=self.n_segments, compactness=self.compactness,
                    sigma=self.sigma
                )
                
                explainer_test.mean_features = trained_mean
                explainer_test.inv_cov_matrices = trained_inv_cov
                explainer_test.top_k = trained_top_k
                explainer_test.n_segments = n_segments_trained

                explainer_test.explain()
                
                pred_unsafe_dict = dict(explainer_test.uncertainty_segments)
                pred_topk_dict = dict(explainer_test.sample_labels_topk)
                
                # Lấy PRED từ giải thích Class 0
                pred_unsafe_c0 = pred_unsafe_dict.get(0, [])
                pred_topk_c0 = pred_topk_dict.get(0, [])
                
                # --- LOGIC MỚI (CHỈ CÓ CLASS 0) ---
                # So sánh PRED (Class 0) với GT (gộp)
                actual_n_segments = len(np.unique(explainer_test.segments_slic))
                p, r, f1, acc = self._calculate_metrics(gt_unsafe_labels, pred_unsafe_c0, actual_n_segments)
                hr = self._calculate_hit_rate(gt_unsafe_labels, pred_topk_c0)
                run_results_c0['precision'].append(p)
                run_results_c0['recall'].append(r)
                run_results_c0['f1'].append(f1)
                run_results_c0['accuracy'].append(acc)
                run_results_c0['hit_rate'].append(hr)
                
                del explainer_test
                torch.cuda.empty_cache()

            # 3.5. Tổng kết ảnh (CHỈ CÓ CLASS 0)
            for metric in all_results_c0.keys():
                mean_val = np.nanmean(run_results_c0[metric])
                all_results_c0[metric].append(mean_val)
            tqdm.write(f"  Ảnh {img_idx} [Giải thích Class 0] - F1 TB: {np.nanmean(run_results_c0['f1']):.3f}, HitRate TB: {np.nanmean(run_results_c0['hit_rate']):.3f}")

        # --- GIAI ĐOẠN 3: BÁO CÁO KẾT QUẢ CUỐI CÙNG ---
        print("\n" + "="*50)
        print("--- Giai đoạn 3: Báo cáo Kết quả Tổng kết (TB ± Lệch chuẩn) ---")
        print("="*50)
        
        # --- ĐỔI TÊN TIÊU ĐỀ (CHỈ CÓ CLASS 0) ---
        print("\n=== Kết quả [Giải thích Class 0 (Cat)] vs [GT Gộp] ===")
        if all_results_c0['f1']:
            for metric in all_results_c0.keys():
                mean_val = np.nanmean(all_results_c0[metric])
                std_val = np.nanstd(all_results_c0[metric])
                print(f"  {metric.upper():<10}: {mean_val:.4f} ± {std_val:.4f}")
        else:
            print("  Không có dữ liệu cho Class 0.")
            
        print("\n" + "="*50)
        print("THÍ NGHIỆM HOÀN TẤT.")


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