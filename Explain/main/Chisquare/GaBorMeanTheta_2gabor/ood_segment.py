import numpy as np
import torch
from skimage.segmentation import slic
# Giả sử KernelExplainer được import từ thư viện SHAP gốc hoặc từ local
from oodxai.Explain import KernelExplainer 
from tqdm.autonotebook import tqdm
import torchvision.transforms as transforms
from PIL import Image
import math
from numpy.linalg import pinv
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from skimage.filters import gabor
from skimage.color import rgb2gray
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
        self.mean_features = None # Sẽ được tính sau
        self.inv_cov_matrices = None # Sẽ được tính sau
        print("-> Ood_segment đã được tạo và cấu hình đúng chuẩn.")
    def get_gabor_energy(self, image_numpy_float, frequency=0.6, theta=0):
        if image_numpy_float.shape[-1] == 3:
            image_gray = rgb2gray(image_numpy_float)
        else:
            image_gray = image_numpy_float
        f_high = 0.6
        f_low = 0.05 
        orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        energy_maps = []
        energy_maps_low = []
        for theta in orientations:
            # Sigma nên tỉ lệ nghịch với tần số để giữ băng thông ổn định
            # sigma ~ 1/f. Ví dụ: f=0.1 -> sigma=3, f=0.35 -> sigma=1
            filt_real, filt_imag = gabor(image_gray, frequency=f_low, theta=theta, sigma_x=3, sigma_y=3)
            energy_maps_low.append(np.sqrt(filt_real**2 + filt_imag**2))
        energy_low = np.mean(np.stack(energy_maps_low), axis=0)

        # --- Tính cho High Frequency ---
        energy_maps_high = []
        for theta in orientations:
            filt_real, filt_imag = gabor(image_gray, frequency=f_high, theta=theta, sigma_x=1, sigma_y=1)
            energy_maps_high.append(np.sqrt(filt_real**2 + filt_imag**2))
        energy_high = np.mean(np.stack(energy_maps_high), axis=0)
        return energy_high,  energy_low
    def Extract(self):
        """
        Trích xuất đặc trưng (phiên bản mở rộng).
        """
        print(f"\n--- Bắt đầu quy trình trích xuất đặc trưng phiên bản top_k tối ưu ---")
        MAX_K_TO_ANALYZE = 10 # Cố định k tối đa = 10 theo yêu cầu
        k_votes = [] # List để lưu phiếu bầu k_j của mỗi ảnh
        temp_results = {class_idx: [] for class_idx in range(self.num_classes)}
        segment_label_topk_nsamples = {class_idx: [] for class_idx in range(self.num_classes)}

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
        for i, image_tensor in enumerate(tqdm(self.background_data, desc="Processing Images")):
            
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
            
            # Xử lí các trường hợp đặc biệt
            if len(positive_shap_indices) == 0:
                num_cols_to_pad = 4 * MAX_K_TO_ANALYZE
                row_data = [i] + [np.nan] * num_cols_to_pad
                temp_results[predicted_class].append(row_data)
                continue
            
            # Chọn ra những segment dương
            positive_shap_values = shap_values_for_predicted_class[positive_shap_indices]
            # Tính sum các value dương để ép về phân phối
            total_positive_shap = np.sum(positive_shap_values)

            # Chọn ra các cặp bao gồm label và giá trị shapely_dương
            shap_label_pairs = list(zip(positive_shap_values, positive_shap_indices))
            # Thực hiện việc sort dựa vào giá trị x[0] là shapely_values
            sorted_by_shap = sorted(shap_label_pairs, key=lambda x: x[0], reverse=True)
            # Chọn ra từ 0 -> k
            sorted_positive_values = [x[0] for x in sorted_by_shap]
            
            if len(sorted_positive_values) <= 1:
                k_votes.append(1) # Bầu cho 1 nếu không đủ so sánh
            
            else:
                # --- 1. Tính k_gap (Logic "Điểm rơi" - Elbow Point) ---
                k_gap = 1 # Mặc định là 1
                values_to_compare_gap = sorted_positive_values[:MAX_K_TO_ANALYZE]
                if len(values_to_compare_gap) > 1:
                    # Tính độ sụt giảm h_k - h_{k+1}
                    drops = [values_to_compare_gap[k] - values_to_compare_gap[k+1] for k in range(len(values_to_compare_gap) - 1)]
                    if drops: # Nếu list drops không rỗng
                        k_j_index = np.argmax(drops)
                        k_gap = k_j_index + 1
                
                # --- 2. Tính k_constraint (Logic "Ràng buộc" - >50% Sum) ---
                k_constraint = 1 # Mặc định là 1
                total_positive_shap = np.sum(sorted_positive_values) # Tổng của TẤT CẢ shap dương
                
                if total_positive_shap > 0:
                    cumulative_sum = 0
                    k_constraint_found = False
                    values_to_compare_constraint = sorted_positive_values[:MAX_K_TO_ANALYZE]
                    
                    for k_index in range(len(values_to_compare_constraint)):
                        cumulative_sum += values_to_compare_constraint[k_index]
                        
                        # Kiểm tra xem tổng tích lũy đã > 50% tổng TOÀN BỘ chưa
                        if cumulative_sum > (0.5 * total_positive_shap):
                            k_constraint = k_index + 1 # Chuyển 0-based index sang 1-based k
                            k_constraint_found = True
                            break # Đã tìm thấy k nhỏ nhất
                    
                    if not k_constraint_found:
                        # Nếu lặp hết top 10 mà vẫn không > 50%
                        # Bầu cho k lớn nhất có thể
                        k_constraint = len(values_to_compare_constraint) if values_to_compare_constraint else 1
                
                # --- 3. Quyết định k_j cuối cùng ---
                # "Luôn tìm gap LỚN NHẤT (k_gap), rồi MỚI xét ràng buộc (k_constraint)"
                # Logic: k_j phải thỏa mãn CẢ HAI, nên ta lấy k lớn hơn.
                k_j = max(k_gap, k_constraint)
                k_votes.append(k_j)
            # --- *** LOGIC MỚI KẾT THÚC TẠI ĐÂY *** ---

            # --- THAY ĐỔI: GIAI ĐOẠN 1, Bước 3b - Trích xuất "dư thừa" ---
            # Trích xuất đặc trưng cho tất cả MAX_K_TO_ANALYZE segments
            segments_to_extract = sorted_by_shap[:MAX_K_TO_ANALYZE]
            gabor_hight, gabor_low = self.get_gabor_energy(image_numpy_float, frequency=0.6, theta=0)
            #  Chuẩn bị các bước tính toán tiếp theo
            top_intensities = []
            top_shap_probs = []
            top_pixel_counts = []
            top_gabor_hight_features = []
            top_gabor_low_features = []
            top_segment_label = []
            
            for shap_values, segment_label in segments_to_extract:
                # prop
                prop = round(shap_values / total_positive_shap, 3) if total_positive_shap > 0 else 0
                top_shap_probs.append(prop)
                # segment laebl
                top_segment_label.append(segment_label)
                # intensity RGB
                mask = (segments_slic == segment_label)
                pixels_in_segment = image_numpy_float[mask]
                pixel_count = pixels_in_segment.shape[0]
                top_pixel_counts.append(pixel_count)

                # Gabor feature
                gabor_high_values = gabor_hight[mask]
                gabor_low_values = gabor_low[mask]
                if gabor_high_values.size > 0:
                    # Lấy trung bình năng lượng Gabor của segment đó:
                    avg_gabor = np.mean(gabor_high_values)
                    top_gabor_hight_features.append(avg_gabor)
                else:
                    top_gabor_hight_features.append(np.nan)
                if gabor_low_values.size > 0:
                    avg_gabor_low = np.mean(gabor_low_values)
                    top_gabor_low_features.append(avg_gabor_low)
                else:
                    top_gabor_low_features.append(np.nan)
                # Pixel Intensity    
                if pixels_in_segment.size > 0:
                    avg_intensity = np.mean(pixels_in_segment) * 255
                    top_intensities.append(avg_intensity)
                else:
                    top_intensities.append(np.nan)
            # Làm đầy dữ liệu
            while len(top_intensities) < MAX_K_TO_ANALYZE:
                top_intensities.append(np.nan)
            while len(top_shap_probs) < MAX_K_TO_ANALYZE:
                top_shap_probs.append(np.nan)
            while len(top_pixel_counts) < MAX_K_TO_ANALYZE:
                top_pixel_counts.append(np.nan)
            while len(top_gabor_hight_features) < MAX_K_TO_ANALYZE:
                top_gabor_hight_features.append(np.nan)
            while len(top_gabor_low_features) < MAX_K_TO_ANALYZE:
                top_gabor_low_features.append(np.nan)
            while len(top_segment_label) < MAX_K_TO_ANALYZE:
                top_segment_label.append(np.nan)

            # Kết hợp dữ liệu thành một hàng
            row_data = [i]  + top_shap_probs + top_intensities + top_pixel_counts + top_gabor_hight_features + top_gabor_low_features
            row_segment = [i] + top_segment_label
            # Lưu kết quả vào đúng vị trí
            temp_results[predicted_class].append(row_data)
            segment_label_topk_nsamples[predicted_class].append(row_segment)
           
        if not k_votes:
            print("!!! Cảnh báo: Không có phiếu bầu nào được ghi nhận. Đặt k_final=1.")
            k_final = 3
        else:
            # Tính toán k_optimal theo yêu cầu: chọn k lớn nhất khi bằng phiếu
            counts = Counter(k_votes)
            max_freq = max(counts.values())
            # Tìm tất cả các k có cùng tần suất cao nhất
            tied_ks = [k for k, freq in counts.items() if freq == max_freq]
            k_optimal = max(tied_ks) # CHỌN K LỚN NHẤT theo yêu cầu
            
            # Tính k_final theo công thức
            k_final = k_optimal 
            
            print(f"\n--- Tính toán Top-K tối ưu ---")
            print(f"Phiếu bầu K (k_j): {counts}")
            print(f"Giá trị K có nhiều phiếu nhất (k_optimal): {k_optimal} (ưu tiên k lớn nhất khi bằng phiếu)")
            print(f"==> K cuối cùng được sử dụng (k_final = k_optimal + 1): {k_final}")

        # Giới hạn k_final không vượt quá MAX_K_TO_ANALYZE
        if k_final > MAX_K_TO_ANALYZE:
            print(f"!!! Cảnh báo: k_final ({k_final}) vượt quá MAX_K_TO_ANALYZE ({MAX_K_TO_ANALYZE}). Giới hạn k_final = {MAX_K_TO_ANALYZE}.")
            k_final = MAX_K_TO_ANALYZE

        result_list = []
        num_full_cols = 1 + 5 * MAX_K_TO_ANALYZE # Số cột đầy đủ
        for class_idx, matrix_list in sorted(temp_results.items()):
            if matrix_list:
                matrix_np = np.array(matrix_list)
                result_list.append((class_idx, matrix_np))
            else:
                result_list.append((class_idx, np.empty((0, num_full_cols))))
        
        # 5b. Cắt tỉa result_list (dữ liệu đặc trưng)
        final_result_list = []
        # Tạo list các chỉ số cột cần giữ
        cols_to_keep = [0] # Cột ID
        cols_to_keep.extend(range(1, k_final + 1)) # Cột shap_prob (từ 1 đến k_final)
        cols_to_keep.extend(range(MAX_K_TO_ANALYZE + 1, MAX_K_TO_ANALYZE + k_final + 1)) # Cột intensity
        cols_to_keep.extend(range(2 * MAX_K_TO_ANALYZE + 1, 2 * MAX_K_TO_ANALYZE + k_final + 1)) # Cột pixel_count
        cols_to_keep.extend(range(3 * MAX_K_TO_ANALYZE + 1, 3 * MAX_K_TO_ANALYZE + k_final + 1)) # Cột gabor_feature
        cols_to_keep.extend(range(4 * MAX_K_TO_ANALYZE + 1, 4 * MAX_K_TO_ANALYZE + k_final + 1))
        for class_idx, matrix_np in result_list:
            if matrix_np.size > 0:
                trimmed_matrix = matrix_np[:, cols_to_keep]
                final_result_list.append((class_idx, trimmed_matrix))
            else:
                # Ma trận rỗng, cần tạo ma trận rỗng mới với đúng số cột đã cắt
                num_final_cols_trimmed = 1 + 5 * k_final
                final_result_list.append((class_idx, np.empty((0, num_final_cols_trimmed))))
        
        # Gán dữ liệu đã cắt tỉa vào thuộc tính của class
        self.aggregated_intensities_tuple = tuple(final_result_list)
        
        # 5c. Cắt tỉa segment_label_topk_nsamples (dữ liệu nhãn)
        final_segment_label_data = {}
        cols_to_keep_labels = [0] # Cột ID
        cols_to_keep_labels.extend(range(1, k_final + 1)) # Cột label (từ 1 đến k_final)
        
        for class_idx, rows_list in segment_label_topk_nsamples.items():
            if rows_list:
                matrix_np_labels = np.array(rows_list)
                # Đảm bảo ma trận không rỗng trước khi cắt
                if matrix_np_labels.size > 0:
                     trimmed_matrix_labels = matrix_np_labels[:, cols_to_keep_labels]
                     final_segment_label_data[class_idx] = trimmed_matrix_labels.tolist() # Chuyển lại thành list of lists
                else:
                     final_segment_label_data[class_idx] = []
            else:
                final_segment_label_data[class_idx] = []
                
        # Ghi đè segment_label_topk_nsamples bằng bản đã cắt tỉa
        segment_label_topk_nsamples = final_segment_label_data # Biến này sẽ được truyền đi
        
        # --- GIAI ĐOẠN 2, Bước 6 - Gọi các hàm con với k_final ---
        
        self.mean(k_final) # <-- THAY ĐỔI: Gọi self.mean với k_final
        
        representative_samples = self.it_should_be_like_that()
        representative_segment_labels = self.get_labels_for_representatives(representative_samples, segment_label_topk_nsamples)
        
        print(f"\n--- Hoàn tất trích xuất đặc trưng. Kết quả có dạng ma trận mở rộng. ---")
        print(self.mean_features)
        
        return self.aggregated_intensities_tuple, representative_segment_labels, self.mean_features, self.inv_cov_matrices, k_final
        
    def mean(self, top_k):
        """
        Tính toán các mô hình thống kê (mean, covariance, inverse covariance)
        cho TỪNG segment trong top_k một cách riêng biệt.

        Sau khi chạy, các thuộc tính sau sẽ được tạo:
        - self.mean_vectors: Dict chứa list các vector trung bình.
        ví dụ: {class_0: [mean_vec_k1, mean_vec_k2, ...], class_1: [...]}
        - self.inv_cov_matrices: Dict chứa list các ma trận hiệp phương sai nghịch đảo.
        ví dụ: {class_0: [inv_cov_k1, inv_cov_k2, ...], class_1: [...]}
        """
        print(f"\n--- Bắt đầu tính toán  mô hình với top_k {top_k} thống kê cho từng bậc segment ---")
        
        # 1. Khởi tạo danh sách kết quả BÊN NGOÀI vòng lặp
        summary_list = []
        inv_cov_summary_list = []
        # Đảm bảo self.aggregated_intensities_tuple được sắp xếp theo class_idx
        # (Hàm Extract của bạn đã làm điều này với sorted(temp_results.items()))
        for class_idx, matrix in self.aggregated_intensities_tuple:
            
            # Tạo một danh sách để lưu các đặc trưng trung bình cho LỚP HIỆN TẠI
            class_features = []
            class_inv_cov_list = []

            if matrix.shape[0] > 3:  # Chỉ tính toán nếu có dữ liệu cho lớp này
                for k in range(top_k):
                    # Lấy các cột thông tin (logic này của bạn đã đúng)
                    shap_prob_col = matrix[:, 1 + k]
                    intensity_col = matrix[:, 1 + top_k + k]
                    pixel_count_col = matrix[:, 1 + 2 * top_k + k]
                    gabor_high_col = matrix[:, 1 + 3 * top_k + k]  # <--- MỚI
                    gabor_low_col = matrix[:, 1 + 4 * top_k + k]
                    # Lấy mean (sử dụng nanmean là rất tốt)
                    # Nó tự động xử lí NaN mà không cần phải thay thế 
                    mean_shap_prob = np.nanmean(shap_prob_col)
                    mean_intensity = np.nanmean(intensity_col)
                    mean_pixel_count = np.nanmean(pixel_count_col)
                    mean_gabor_high = np.nanmean(gabor_high_col)   # <--- MỚI
                    mean_gabor_low = np.nanmean(gabor_low_col) # Mới
                    # Nối dài danh sách đặc trưng cho lớp hiện tại
                    class_features.append([mean_shap_prob, mean_intensity, mean_pixel_count, mean_gabor_high, mean_gabor_low])

                    # --- BỔ SUNG: Tính toán ma trận hiệp phương sai cho k hiện tại ---
                    # 1. Ghép 4 cột đặc trưng lại thành một ma trận (số_ảnh, 4)
                    feature_matrix_k = np.stack([shap_prob_col, intensity_col, pixel_count_col, gabor_high_col, gabor_low_col], axis=1)

                    # 1. Tính trung bình của mỗi cột, bỏ qua các giá trị NaN có sẵn
                    col_mean_k = np.nanmean(feature_matrix_k, axis=0)

                    # 2. Tìm vị trí (chỉ số) của tất cả các giá trị NaN
                    nan_indices_k = np.where(np.isnan(feature_matrix_k))

                    # 3. Tại các vị trí đó, thay thế NaN bằng giá trị trung bình của cột tương ứng
                    feature_matrix_k[nan_indices_k] = np.take(col_mean_k, nan_indices_k[1])
                    # np.cov() để tính ma trận hiệp phương sai
                    # rowvar = False để mỗi cột là 1 biến mỗi hàng là 1 mẫu (quan sát)
                    # np.identity(3) * 1e-6 để tránh ma trận suy biến hay còn gọi là Regularization 
                    # Mục đích là khiến ma trận hiệp phương sai không bị đặc, tránh lỗi khi tính nghịch đảo
                    # Hay nói các khác là +1 giá trị 1*1.10^-6 vào đường chéo chính
                    # Từ đó ma trận luôn có thể nghịch đảo
                    cov_matrix_k = np.cov(feature_matrix_k, rowvar=False) + np.identity(5) * 1e-6

                    # Ma trận thực tế rất có thể bị suy biến, nên ta dùng pseudo-inverse
                    # Ma trận suy biến DET = 0 
                    inv_cov_matrix_k = pinv(cov_matrix_k)
                    
                    # 4. Thêm ma trận vừa tính vào list của lớp hiện tại
                    class_inv_cov_list.append(inv_cov_matrix_k)

            else:
                class_features = np.full((top_k, 5), np.nan).tolist()
                # --- BỔ SUNG: Điền ma trận rỗng nếu không đủ dữ liệu ---
                class_inv_cov_list = np.full((top_k, 5, 5), np.nan)
                
            # 2. Thêm kết quả của lớp này vào danh sách tổng
            summary_list.append(class_features)
            inv_cov_summary_list.append(class_inv_cov_list)
        
        # 3. Chuyển đổi danh sách các danh sách thành ma trận NumPy SAU KHI vòng lặp kết thúc
        self.inv_cov_matrices = np.array(inv_cov_summary_list)
        self.mean_features = np.array(summary_list)
        print('Hoàn thành việc tính mean vector')
        
        # Lưu kết quả vào một thuộc tính của class nếu cần dùng lại
    def it_should_be_like_that(self):
        representative_samples = {}
        for (class_idx, data_matrix), mean_vector_3d in zip(self.aggregated_intensities_tuple,self.mean_features):
            # Ép về 1 vecotr bao gồm [mean_shap_prob_1, mean_intensity_1, mean_pixel_count_1] * top_k
            # shape (top_k*3 loại thông tin, )
            mean_vector_flat  = mean_vector_3d.reshape(-1)
            # Lấy các vector đặc trưng từ ma trận dữ liệu -> Bỏ cái index 1 từ từ tính sau
            feature_vectors = data_matrix[:, 1:]      
            # Xử lí với NaN -> ánh xạ về 0 để không ảnh hưởng đến sự đóng góp.
            mean_vector_flat = np.nan_to_num(mean_vector_flat)
            feature_vectors = np.nan_to_num(feature_vectors)  
        
            # Similarity Cosine
            # Reshpae phát nữa thành (1,9) để hàm cosine_similarity hiểu
            # shape (class, độ tương đồng  * các samples)
            similarities = cosine_similarity(mean_vector_flat.reshape(1, -1), feature_vectors)

            similarities_1d = similarities[0]
            best_match_index = np.argmax(similarities_1d)

            original_image_id = int(data_matrix[best_match_index, 0])
            max_similarity_score = similarities_1d[best_match_index]
            
            representative_samples[class_idx] = {
                'representative_id': original_image_id,
                'similarity_score': max_similarity_score
            }
            
            # print(f"Lớp {class_idx}: Mẫu đại diện là ảnh ID {original_image_id} (độ tương đồng: {max_similarity_score:.4f})")
        return representative_samples
    def get_labels_for_representatives(self, representative_samples, segment_label_data):
        """
        Từ ID của các ảnh đại diện, tra cứu trong dữ liệu segment_label_topk_nsamples
        để trả về hàng thông tin đầy đủ [i, label_1, label_2, ...].

        Args:
            representative_samples (dict): Dictionary chứa ID ảnh đại diện cho mỗi lớp.
            segment_label_data (dict): Dictionary chứa list các hàng label cho mỗi lớp.

        Returns:
            dict: Một dictionary ánh xạ class_idx tới hàng thông tin của ảnh đại diện.
        """
        print("\n--- Bắt đầu tra cứu segment labels cho các mẫu đại diện ---")
        results = {}
        
        # Lặp qua từng lớp và thông tin ảnh đại diện của nó
        for class_idx, info in representative_samples.items():
            target_id = info['representative_id']
            
            # Kiểm tra xem lớp này có trong dữ liệu label không
            if class_idx in segment_label_data:
                # Lấy list tất cả các hàng label của lớp này
                all_rows_for_class = segment_label_data[class_idx]
                
                # Dò tìm trong list đó để tìm hàng có ID khớp
                found_row = None
                for row in all_rows_for_class:
                    if row[0] == target_id:
                        found_row = row
                        break # Tìm thấy rồi thì dừng vòng lặp
                
                results[class_idx] = found_row
            else:
                results[class_idx] = None # Nếu lớp không tồn tại trong dữ liệu label

        return results


    def extract_and_save_statistics(self, output_filename=None):
        """
        Thực hiện quy trình trích xuất đặc trưng đầy đủ (tương tự Extract)
        nhưng KHÔNG tính toán mẫu đại diện.
        
        Thay vào đó, nó sẽ lưu các mô hình thống kê quan trọng
        (k_final, mean_features, inv_cov_matrices) vào một file .txt.
        """
        print(f"\n--- Bắt đầu quy trình trích xuất và lưu thống kê ---")
        
        # Đặt tên file output mặc định nếu không được cung cấp
        if output_filename is None:
            output_filename = f"model_stats_{self.Ood_name or 'default'}.txt"

        # --- GIAI ĐOẠN 1: TRÍCH XUẤT (Tương tự hàm Extract) ---
        MAX_K_TO_ANALYZE = 10 # Cố định k tối đa = 10
        k_votes = [] # List để lưu phiếu bầu k_j của mỗi ảnh
        temp_results = {class_idx: [] for class_idx in range(self.num_classes)}
        segment_label_topk_nsamples = {class_idx: [] for class_idx in range(self.num_classes)}

        # 1. HÀM HELPER: Giải chuẩn hóa về numpy float [0, 1]
        def get_float_image_from_tensor(tensor_image):
            std = np.array(self.transform_std, dtype=np.float32)
            mean = np.array(self.transform_mean, dtype=np.float32)
            np_image = tensor_image.cpu().numpy().transpose((1, 2, 0))
            np_image = std * np_image + mean
            return np.clip(np_image, 0, 1)

        print(f"Bắt đầu xử lý {len(self.background_data)} ảnh nền...")
        for i, image_tensor in enumerate(tqdm(self.background_data, desc="Processing Images for Stats")):
            
            with torch.no_grad():
                logits = self.model(image_tensor.unsqueeze(0).to(self.device))
                predicted_class = torch.argmax(logits, dim=1).item()

            # 2. CHUẨN BỊ ẢNH VÀ PHÂN MẢNH
            image_numpy_float = get_float_image_from_tensor(image_tensor)
            segments_slic = slic(image_numpy_float, n_segments=self.n_segments,
                                compactness=self.compactness, sigma=self.sigma, start_label=self.start_label)
            num_actual_superpixels = len(np.unique(segments_slic))
            background_color = image_numpy_float.mean((0, 1))

            transform_for_prediction = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((224, 224)),
                    transforms.Normalize(self.transform_mean, self.transform_std)
                ])
            
            # 3. PREDICTION_FUNCTION "VECTOR HÓA"
            def prediction_function(z):
                batch_size = 10
                all_logits = []
                h, w, c = image_numpy_float.shape
                unique_labels = np.unique(segments_slic)
                for i in range(0, z.shape[0], batch_size):
                    z_batch = z[i:i + batch_size]
                    masked_images_np = []
                    
                    for mask in z_batch:
                        temp_image = image_numpy_float.copy()
                        inactive_segments = np.where(mask == 0)[0]
                        inactive_labels = unique_labels[inactive_segments]
                        mask_all_inactive = np.isin(segments_slic, inactive_labels)
                        temp_image[mask_all_inactive] = background_color
                        masked_images_np.append(temp_image)
                    
                    tensors = torch.stack(
                        [transform_for_prediction(img) for img in masked_images_np]
                    ).to(self.device)

                    with torch.no_grad():
                        logits_shap = self.model(tensors)
                    all_logits.append(logits_shap.cpu().numpy())
                return np.concatenate(all_logits, axis=0)
            
            # 4. CHẠY SHAP
            explainer = KernelExplainer(prediction_function, np.zeros((1, num_actual_superpixels)))
            shap_values_for_image = explainer.shap_values(np.ones((1, num_actual_superpixels)), nsamples=self.num_samples)

            # 5. TÍNH TOÁN K_J (Top-K cho ảnh)
            shap_values_for_predicted_class = shap_values_for_image[0, :, predicted_class]
            positive_shap_indices = np.where(shap_values_for_predicted_class > 0)[0]
            
            if len(positive_shap_indices) == 0:
                num_cols_to_pad = 5 * MAX_K_TO_ANALYZE
                row_data = [i] + [np.nan] * num_cols_to_pad
                temp_results[predicted_class].append(row_data)
                continue
            
            positive_shap_values = shap_values_for_predicted_class[positive_shap_indices]
            shap_label_pairs = list(zip(positive_shap_values, positive_shap_indices))
            sorted_by_shap = sorted(shap_label_pairs, key=lambda x: x[0], reverse=True)
            sorted_positive_values = [x[0] for x in sorted_by_shap]
            
            if len(sorted_positive_values) <= 1:
                k_votes.append(1)
            else:
                # k_gap
                k_gap = 1
                values_to_compare_gap = sorted_positive_values[:MAX_K_TO_ANALYZE]
                if len(values_to_compare_gap) > 1:
                    drops = [values_to_compare_gap[k] - values_to_compare_gap[k+1] for k in range(len(values_to_compare_gap) - 1)]
                    if drops:
                        k_j_index = np.argmax(drops)
                        k_gap = k_j_index + 1
                
                # k_constraint
                k_constraint = 1
                total_positive_shap = np.sum(sorted_positive_values)
                if total_positive_shap > 0:
                    cumulative_sum = 0
                    k_constraint_found = False
                    values_to_compare_constraint = sorted_positive_values[:MAX_K_TO_ANALYZE]
                    
                    for k_index in range(len(values_to_compare_constraint)):
                        cumulative_sum += values_to_compare_constraint[k_index]
                        if cumulative_sum > (0.5 * total_positive_shap):
                            k_constraint = k_index + 1
                            k_constraint_found = True
                            break
                    if not k_constraint_found:
                        k_constraint = len(values_to_compare_constraint) if values_to_compare_constraint else 1
                
                # Quyết định k_j
                k_j = max(k_gap, k_constraint)
                k_votes.append(k_j)

            # 6. TRÍCH XUẤT ĐẶC TRƯNG "DƯ THỪA" (CHO MAX_K)
            gabor_high, gabor_low = self.get_gabor_energy(image_numpy_float, frequency=0.6, theta=0)
            segments_to_extract = sorted_by_shap[:MAX_K_TO_ANALYZE]
            top_intensities, top_shap_probs, top_pixel_counts, top_gabor_high_features, top_gabor_low_features, top_segment_label = [], [], [], [], [], []
            
            total_positive_shap = np.sum(positive_shap_values) # Tính lại tổng SHAP dương
            
            for shap_values, segment_label in segments_to_extract:
                prop = round(shap_values / total_positive_shap, 3) if total_positive_shap > 0 else 0
                top_shap_probs.append(prop)
                top_segment_label.append(segment_label)
                
                mask = (segments_slic == segment_label)
                pixels_in_segment = image_numpy_float[mask]
                pixel_count = pixels_in_segment.shape[0]
                top_pixel_counts.append(pixel_count)
                gabor_high_vals = gabor_high[mask]
                gabor_low_vals = gabor_low[mask]
                #Gabor
                if gabor_high_vals.size > 0:
                    top_gabor_high_features.append(np.mean(gabor_high_vals))
                else:
                    top_gabor_high_features.append(np.nan)
                if gabor_low_vals.size > 0:
                    top_gabor_low_features.append(np.mean(gabor_low_vals))
                else:
                    top_gabor_low_features.append(np.nan)
                if pixels_in_segment.size > 0:
                    avg_intensity = np.mean(pixels_in_segment) * 255
                    top_intensities.append(avg_intensity)
                else:
                    top_intensities.append(np.nan)

            # Làm đầy dữ liệu
            while len(top_intensities) < MAX_K_TO_ANALYZE: top_intensities.append(np.nan)
            while len(top_shap_probs) < MAX_K_TO_ANALYZE: top_shap_probs.append(np.nan)
            while len(top_pixel_counts) < MAX_K_TO_ANALYZE: top_pixel_counts.append(np.nan)
            while len(top_gabor_high_features) < MAX_K_TO_ANALYZE: top_gabor_high_features.append(np.nan)
            while len(top_gabor_low_features) < MAX_K_TO_ANALYZE: top_gabor_low_features.append(np.nan)
            row_data = [i]  + top_shap_probs + top_intensities + top_pixel_counts + top_gabor_high_features + top_gabor_low_features
            temp_results[predicted_class].append(row_data)
           
        # --- GIAI ĐOẠN 2: TÍNH TOÁN K_FINAL VÀ CẮT TỈA ---
        if not k_votes:
            print("!!! Cảnh báo: Không có phiếu bầu nào. Đặt k_final=3.")
            k_final = 3
        else:
            counts = Counter(k_votes)
            max_freq = max(counts.values())
            tied_ks = [k for k, freq in counts.items() if freq == max_freq]
            k_optimal = max(tied_ks)
            k_final = k_optimal + 1
            print(f"\n--- Tính toán Top-K tối ưu ---")
            print(f"Phiếu bầu K (k_j): {counts}")
            print(f"Giá trị K có nhiều phiếu nhất (k_optimal): {k_optimal}")
            print(f"==> K cuối cùng được sử dụng (k_final = k_optimal + 1): {k_final}")

        if k_final > MAX_K_TO_ANALYZE:
            print(f"!!! Cảnh báo: k_final ({k_final}) vượt quá MAX_K_TO_ANALYZE. Giới hạn k_final = {MAX_K_TO_ANALYZE}.")
            k_final = MAX_K_TO_ANALYZE

        # Sắp xếp và Cắt tỉa result_list (dữ liệu đặc trưng)
        result_list = []
        num_full_cols = 1 + 5 * MAX_K_TO_ANALYZE
        for class_idx, matrix_list in sorted(temp_results.items()):
            if matrix_list:
                result_list.append((class_idx, np.array(matrix_list)))
            else:
                result_list.append((class_idx, np.empty((0, num_full_cols))))
        
        final_result_list = []
        cols_to_keep = [0] # Cột ID
        cols_to_keep.extend(range(1, k_final + 1)) # Cột shap_prob
        cols_to_keep.extend(range(MAX_K_TO_ANALYZE + 1, MAX_K_TO_ANALYZE + k_final + 1)) # Cột intensity
        cols_to_keep.extend(range(2 * MAX_K_TO_ANALYZE + 1, 2 * MAX_K_TO_ANALYZE + k_final + 1)) # Cột pixel_count
        cols_to_keep.extend(range(3 * MAX_K_TO_ANALYZE + 1, 3 * MAX_K_TO_ANALYZE + k_final + 1)) # Cột pixel_count
        cols_to_keep.extend(range(4 * MAX_K_TO_ANALYZE + 1, 4 * MAX_K_TO_ANALYZE + k_final + 1)) # Cột pixel_count

        for class_idx, matrix_np in result_list:
            if matrix_np.size > 0:
                final_result_list.append((class_idx, matrix_np[:, cols_to_keep]))
            else:
                num_final_cols_trimmed = 1 + 5 * k_final
                final_result_list.append((class_idx, np.empty((0, num_final_cols_trimmed))))
        
        self.aggregated_intensities_tuple = tuple(final_result_list)
        
        # --- GIAI ĐOẠN 3: TÍNH TOÁN THỐNG KÊ ---
        
        # Gọi hàm mean để tính toán và gán self.mean_features, self.inv_cov_matrices
        self.mean(k_final) 
        
        print(f"\n--- Hoàn tất trích xuất đặc trưng (bỏ qua mẫu đại diện) ---")
        print(f"k_final được xác định là: {k_final}")
        print(f"mean_features shape: {self.mean_features.shape}")
        print(f"inv_cov_matrices shape: {self.inv_cov_matrices.shape}")

        # --- GIAI ĐOẠN 4: LƯU FILE THEO YÊU CẦU ---
        
        print(f"Đang lưu kết quả vào file: {output_filename}...")
        
        if output_filename is None:
            output_filename = f"model_stats_{self.Ood_name or 'default'}.npz"
        elif not output_filename.endswith('.npz'):
            output_filename = output_filename.rsplit('.', 1)[0] + '.npz'
            
        print(f"Đang lưu kết quả tối ưu vào file: {output_filename}...")

        try:
            # Sử dụng np.savez_compressed để lưu nhiều array vào 1 file
            # Nó nhanh, nén, và giữ nguyên kiểu dữ liệu/shape
            np.savez_compressed(
                output_filename, 
                k_final=k_final,  # Lưu k_final (số nguyên)
                mean_features=self.mean_features, # Lưu array 3D
                inv_cov_matrices=self.inv_cov_matrices # Lưu array 4D
            )
            
            print(f"Đã lưu file thành công: {output_filename}")
            
            # Trả về các giá trị đã tính, phòng trường hợp bạn muốn dùng
            return k_final, self.mean_features, self.inv_cov_matrices

        except Exception as e:
            print(f"ĐÃ XẢY RA LỖI trong quá trình lưu file .npz: {e}")
            import traceback
            traceback.print_exc()
            # Vẫn trả về giá trị dù lưu file lỗi
            return k_final, self.mean_features, self.inv_cov_matrices

                    # Vẫn trả về giá trị dù lưu file lỗi

 