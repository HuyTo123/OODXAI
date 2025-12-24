import os
# Ép NumPy và các thư viện tính toán chỉ chạy 1 luồng duy nhất
# Điều này giúp tránh deadlock và thường làm code chạy ổn định hơn trong vòng lặp phức tạp

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
import json
from pathlib import Path
import traceback
import gc
class Ood_segment(OODExplainerBase):
    def __init__(self, model=None, Ood_name=None, background_data=None, sample=None, device=None, class_name=None,
                 n_segments=50, compactness=10, sigma=1, start_label=0, transform_mean=(0.485, 0.456, 0.406), transform_std=(0.229, 0.224, 0.225),
                 num_samples=100):
        """
        Hàm khởi tạo "chuẩn OOP".
        """
        print('n_jop = 1')

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
        self.empirical_distances = None
        print("OOD_SEGMENT has been initialized")
    def get_gabor_energy(self, image_numpy_float):
        if image_numpy_float.shape[-1] == 3:
            image_gray = rgb2gray(image_numpy_float)
        else:
            image_gray = image_numpy_float
        f_high = 0.6
        f_low = 0.05 
        orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]
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
    def _load_ground_truth_staticscal(self, npz_file_path, txt_file_path):
        """
        (Helper) Load toàn bộ thông số đã train:
        1. Load Mean, Cov, k_final, Representative Labels từ .npz
        2. Load Empirical Distances từ .txt (JSON format)
        """
        print(f"\n Loading Ground Truth Statistical Data...")
        
        # 1. Xử lý đường dẫn
        script_dir = Path(__file__).parent
        # Nếu đường dẫn là tương đối, nối với script_dir, nếu tuyệt đối thì giữ nguyên
        full_npz_path = script_dir / npz_file_path if not Path(npz_file_path).is_absolute() else Path(npz_file_path)
        full_txt_path = script_dir / txt_file_path if not Path(txt_file_path).is_absolute() else Path(txt_file_path)

        # ---------------------------------------------------------
        # PHẦN A: Load file .NPZ
        # ---------------------------------------------------------
        try:
            data = np.load(full_npz_path, allow_pickle=True)
            
            # Load các chỉ số cơ bản
            self.top_k = data['k_final'].item() 
            self.mean_features = data['mean_features']
            self.inv_cov_matrices = data['inv_cov_matrices']
            
            # [QUAN TRỌNG] Load thông tin mẫu đại diện (đã lưu ở bước Train)
            # Để không phải tính lại hàm it_should_be_like_that()
            if 'representative_labels' in data:
                self.representative_segment_labels = data['representative_labels'].item()
            else:
                self.representative_segment_labels = None

            data.close()
            print(f"   + Load thành công NPZ. k_final={self.top_k}")

        except Exception as e:
            print(f"!!! LỖI khi đọc file .npz: {e}")
            traceback.print_exc()
            return False # Báo thất bại

        # ---------------------------------------------------------
        # PHẦN B: Load file .TXT (Empirical Distances)
        # ---------------------------------------------------------
        try:
            print(f"-> Đang đọc file Distance (TXT/JSON): {full_txt_path}")
            with open(full_txt_path, 'r') as f:
                # Load JSON: Key là Class ID (string), Value là Dict {Top_k: List[Distances]}
                raw_distances = json.load(f)
            
            # Chuyển đổi key từ String sang Int (do JSON luôn lưu key là string)
            self.empirical_distances = {}
            for class_id_str, k_dict in raw_distances.items():
                class_id = int(class_id_str)
                self.empirical_distances[class_id] = {}
                for k_rank_str, dist_list in k_dict.items():
                    self.empirical_distances[class_id][int(k_rank_str)] = dist_list
            
            print(f"   + Load thành công Distances cho {len(self.empirical_distances)} classes.")

        except FileNotFoundError:
            print(f"!!! LỖI: Không tìm thấy file txt: {full_txt_path}")
            return False
        except json.JSONDecodeError:
            print(f"!!! LỖI: File txt không đúng định dạng JSON.")
            return False
        except Exception as e:
            print(f"!!! LỖI khi đọc file txt: {e}")
            return False

        return True # Thành công toàn bộ

    def Extract(self, npz_path='test/model_stats_MSP.npz', txt_path='test/empirical_distances.txt'):
        """
        Quy trình Extract (Chế độ Load):
        Chỉ tải dữ liệu, KHÔNG tính toán lại.
        """
        # Gọi hàm helper để load dữ liệu
        success = self._load_ground_truth_staticscal(npz_path, txt_path)
        
        if not success:
            print("Quá trình Extract thất bại do lỗi load file.")
            return  None, None, None, None

        print(f"\n--- Extract Mode ---")
        # Trả về các giá trị đã load vào thuộc tính của self
        # Lưu ý: self.empirical_distances được trả về thêm ở cuối
        return  self.mean_features, self.inv_cov_matrices, self.top_k, self.empirical_distances

        
    def mean(self, top_k):
        """
            Calcute the mean feature vector for each class and top_k segments
        """
        print("Mean and covariance calculation is starting...")
        
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
                    col_mean_k = np.nan_to_num(col_mean_k, nan=0.0)

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
        print('Mean  and covariance calculation is completed.')
        
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
    def generate_binary_masks(self, representative_samples, segment_label_topk_nsamples, k_final):
        """
        [ĐÃ SỬA] 
        Hàm này bây giờ sẽ:
        1. Load ảnh gốc.
        2. CHẠY LẠI SLIC để lấy map phân đoạn.
        3. Chỉ giữ lại các pixel thuộc k_final segments quan trọng nhất (Masked Image).
        4. Lưu ảnh Masked đó vào folder represent/{class_name}.
        """
        print(f"\n-> Đang tạo và lưu ảnh Masked (Top {k_final} segments)...")
        final_representative_dict = {}

        # 1. Tạo folder gốc
        current_dir = Path(__file__).resolve().parent

        # 2. Nối thêm folder "represent" vào
        base_save_path = current_dir / "represent"

        # 3. Tạo folder
        base_save_path.mkdir(parents=True, exist_ok=True)

        for class_idx, info in representative_samples.items():
            rep_id = info['representative_id']
            similarity = info['similarity_score']

            # Tìm dòng dữ liệu chứa labels: [Image_ID, Label_1, Label_2, ..., Label_Max]
            labels_row = next((row for row in segment_label_topk_nsamples[class_idx] if int(row[0]) == rep_id), None)

            if labels_row:
                try:
                    # --- BƯỚC A: CHUẨN BỊ ẢNH GỐC (FLOAT) ---
                    img_tensor = self.background_data[rep_id]
                    
                    # Un-normalize về dạng numpy float [0, 1] để chạy SLIC
                    img_numpy = img_tensor.cpu().numpy().transpose((1, 2, 0)) # C,H,W -> H,W,C
                    img_numpy = img_numpy * self.transform_std + self.transform_mean
                    img_numpy_float = np.clip(img_numpy, 0, 1) # Ảnh gốc đầy đủ màu sắc

                    # --- BƯỚC B: CHẠY LẠI SLIC (BẮT BUỘC) ---
                    # Phải dùng đúng tham số như lúc Extract để ra đúng segment ID
                    segments_slic = slic(img_numpy_float, n_segments=self.n_segments,
                                         compactness=self.compactness, sigma=self.sigma, 
                                         start_label=self.start_label)

                    # --- BƯỚC C: TẠO MASKED IMAGE ---
                    # 1. Tạo một tấm ảnh nền đen hoàn toàn
                    masked_image = np.zeros_like(img_numpy_float)

                    # 2. Lấy danh sách k_final labels quan trọng nhất
                    # labels_row[0] là ID ảnh, nên labels bắt đầu từ index 1
                    # Cần lấy k_final labels đầu tiên
                    important_labels = labels_row[1 : 1 + k_final]

                    # 3. "Bật" các pixel thuộc các label quan trọng
                    for label in important_labels:
                        label = int(label)
                        if label == -1: continue # Bỏ qua padding nếu có
                        
                        # Tạo mặt nạ cho label hiện tại
                        mask = (segments_slic == label)
                        
                        # Copy pixel từ ảnh gốc sang ảnh nền đen tại vị trí mask
                        masked_image[mask] = img_numpy_float[mask]

                    # --- BƯỚC D: LƯU ẢNH ---
                    if self.class_name and class_idx < len(self.class_name):
                         cls_name = str(self.class_name[class_idx])
                    else:
                         cls_name = f"class_{class_idx}"
                    
                    class_save_path = base_save_path / cls_name
                    class_save_path.mkdir(parents=True, exist_ok=True)

                    # Chuyển sang uint8 [0, 255] để lưu
                    img_to_save = (masked_image * 255).astype(np.uint8)
                    pil_img = Image.fromarray(img_to_save)

                    file_name = f"rep_id_{rep_id}_sim_{similarity:.3f}_k{k_final}.png"
                    save_path = class_save_path / file_name
                    pil_img.save(save_path)
                    
                    # Lưu thông tin để ghi file .npz
                    final_representative_dict[class_idx] = labels_row

                except Exception as e:
                    print(f"!!! Lỗi khi xử lý ảnh đại diện class {class_idx}: {e}")
                    traceback.print_exc()
            else:
                print(f"!!! Cảnh báo: Không tìm thấy labels cho mẫu đại diện ID {rep_id}")

        return final_representative_dict
    def calculate_and_save_empirical_distances(self, k_final, txt_filename):
        """
        [QUAN TRỌNG] Hàm mới: Tính lại khoảng cách Mahalanobis cho TẤT CẢ mẫu huấn luyện
        và lưu danh sách đã sắp xếp vào file .txt (JSON format).
        """
        print(f"\n--- [Step 3] Tính toán khoảng cách thực nghiệm (Empirical Distances) ---")
        
        empirical_data = {} # Dict để lưu kết quả

        # Duyệt qua từng class
        for class_idx, matrix in self.aggregated_intensities_tuple:
            
            # Khởi tạo dict cho class này
            empirical_data[int(class_idx)] = {} 
            
            mean_vectors_class = self.mean_features[class_idx]
            inv_cov_matrices_class = self.inv_cov_matrices[class_idx]
            
            # Nếu class này không có dữ liệu hợp lệ (toàn NaN), bỏ qua
            if np.isnan(mean_vectors_class).all():
                continue

            # Duyệt qua từng bậc K (từ 0 đến k_final - 1)
            for k in range(k_final):
                distances = []
                
                mean_vec_k = mean_vectors_class[k]
                inv_cov_k = inv_cov_matrices_class[k]
                
                # Lấy dữ liệu raw từ ma trận (đã trích xuất ở Step 1)
                # Cấu trúc cột: [ID, Shap*K, Int*K, Pix*K, GabH*K, GabL*K]
                # Lấy đúng cột cho bậc k
                feature_matrix_k = np.stack([
                    matrix[:, 1 + k],                  # SHAP Prob
                    matrix[:, 1 + k_final + k],        # Intensity
                    matrix[:, 1 + 2 * k_final + k],    # Pixel Count
                    matrix[:, 1 + 3 * k_final + k],    # Gabor High
                    matrix[:, 1 + 4 * k_final + k]     # Gabor Low
                ], axis=1)
                
                # Duyệt qua từng mẫu (sample) trong tập huấn luyện
                for i in range(feature_matrix_k.shape[0]):
                    vec = feature_matrix_k[i]
                    
                    # Nếu vector có NaN -> Bỏ qua mẫu này
                    if np.isnan(vec).any():
                        continue
                        
                    # Tính Mahalanobis Distance: (x-u)T * Cov^-1 * (x-u)
                    diff = vec - mean_vec_k
                    dist_sq = diff.T @ inv_cov_k @ diff
                    distances.append(float(dist_sq)) # Lưu float chuẩn python
                
                # Sắp xếp từ bé đến lớn
                distances.sort()
                
                # Lưu vào dict: Top_K -> List[Distances]
                empirical_data[int(class_idx)][int(k)] = distances
                
        # --- Lưu ra file .txt ---
        try:
            with open(txt_filename, 'w') as f:
                # Dùng JSON để dump structure, dễ đọc cho cả người và máy
                # indent=4 giúp xuống dòng đẹp mắt
                json.dump(empirical_data, f, indent=4)
            print(f" {txt_filename} hasbeen saved successfully.")
        except Exception as e:
            print(f"{txt_filename} saving failed. Error: {e}")
    def extract_and_save_statistics(self, output_filename=None):
        """
        Phiên bản SỬA LỖI: Thêm kiểm tra an toàn cho các trường hợp ảnh bị lỗi segmentation.
        """
        print(f"\n--- Bắt đầu quy trình trích xuất và lưu thống kê ---")
        
        if output_filename is None:
            output_filename = f"model_stats_{self.Ood_name or 'default'}.txt"

        # --- GIAI ĐOẠN 1: TRÍCH XUẤT ---
        MAX_K_TO_ANALYZE = 10 
        k_votes = [] 
        temp_results = {class_idx: [] for class_idx in range(self.num_classes)}
        segment_label_topk_nsamples = {class_idx: [] for class_idx in range(self.num_classes)}

        # Hàm helper lấy ảnh float
        def get_float_image_from_tensor(tensor_image):
            std = np.array(self.transform_std, dtype=np.float32)
            mean = np.array(self.transform_mean, dtype=np.float32)
            np_image = tensor_image.cpu().numpy().transpose((1, 2, 0))
            np_image = std * np_image + mean
            return np.clip(np_image, 0, 1)

        print(f"Bắt đầu xử lý {len(self.background_data)} ảnh nền...")
        
        # Thêm try-except cho vòng lặp chính để không dừng chương trình nếu 1 ảnh lỗi
        valid_images_count = 0
        
        for i, image_tensor in enumerate(tqdm(self.background_data, desc="Processing images", mininterval=1.0)):            
            try:
                with torch.no_grad():
                    logits = self.model(image_tensor.unsqueeze(0).to(self.device))
                    predicted_class = torch.argmax(logits, dim=1).item()

                # 2. CHUẨN BỊ ẢNH VÀ PHÂN MẢNH
                image_numpy_float = get_float_image_from_tensor(image_tensor)
                
                # Phân đoạn ảnh
                segments_slic = slic(image_numpy_float, n_segments=self.n_segments,
                                     compactness=self.compactness, sigma=self.sigma, start_label=self.start_label)
                
                # [FIX QUAN TRỌNG 1] Kiểm tra số lượng segment thực tế
                unique_segments = np.unique(segments_slic)
                num_actual_superpixels = len(unique_segments)
                
                if num_actual_superpixels == 0:
                    print(f"\n[Warning] Ảnh index {i} không tạo được segment nào. Bỏ qua.")
                    continue
                
                # Nếu ảnh quá đơn điệu, SLIC có thể chỉ trả về 1 segment -> SHAP đôi khi gặp lỗi với M=1
                # Nhưng thường M=1 vẫn chạy được, lỗi chủ yếu là M=0.

                background_color = image_numpy_float.mean((0, 1))

                transform_for_prediction = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Resize((224, 224)),
                        transforms.Normalize(self.transform_mean, self.transform_std)
                    ])
                
                # 3. PREDICTION_FUNCTION AN TOÀN
                # 3. PREDICTION_FUNCTION (Phiên bản Debug "Mổ xẻ")
                def prediction_function(z):
                    batch_size = 20
                    all_logits = []
                    unique_labels = np.unique(segments_slic)
                    # Thêm leave=False để tqdm không làm lộn xộn output của vòng lặp ngoài
                    for i in tqdm(range(0, z.shape[0], batch_size), desc="SHAP Batches", leave=False):
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
                                
                        self.model.eval()
                        with torch.no_grad():
                            logits = self.model(tensors)
                        all_logits.append(logits)
                    
                    all_logits_numpy = [l.cpu().numpy() for l in all_logits]
                    return np.concatenate(all_logits_numpy, axis=0)
                
                # 4. CHẠY SHAP
                explainer = KernelExplainer(prediction_function, np.zeros((1, num_actual_superpixels)))
                
                # Tính nsamples, đảm bảo không quá nhỏ
                # nsamples_cal = 1000 - checking param
                print('giá trị  ', num_actual_superpixels)
                # Chạy SHAP
                shap_values_for_image = explainer.shap_values(np.ones((1, num_actual_superpixels)), nsamples= 'auto',gc_collect=True, n_jobs=1)

                # 5. XỬ LÝ KẾT QUẢ (Logic cũ giữ nguyên)
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
                
                # --- Logic tính k_votes (Giữ nguyên) ---
                if len(sorted_positive_values) <= 1:
                    k_votes.append(1)
                else:
                    k_gap = 1
                    values_to_compare_gap = sorted_positive_values[:MAX_K_TO_ANALYZE]
                    if len(values_to_compare_gap) > 1:
                        drops = [values_to_compare_gap[k] - values_to_compare_gap[k+1] for k in range(len(values_to_compare_gap) - 1)]
                        if drops:
                            k_j_index = np.argmax(drops)
                            k_gap = k_j_index + 1
                    
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
                    
                    k_j = max(k_gap, k_constraint)
                    k_votes.append(k_j)

                # --- Trích xuất đặc trưng dư thừa (Giữ nguyên) ---
                gabor_high, gabor_low = self.get_gabor_energy(image_numpy_float)
                segments_to_extract = sorted_by_shap[:MAX_K_TO_ANALYZE]
                top_intensities, top_shap_probs, top_pixel_counts, top_gabor_high_features, top_gabor_low_features, top_segment_label = [], [], [], [], [], []
                
                total_positive_shap = np.sum(positive_shap_values)
                
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

                while len(top_intensities) < MAX_K_TO_ANALYZE: top_intensities.append(np.nan)
                while len(top_shap_probs) < MAX_K_TO_ANALYZE: top_shap_probs.append(np.nan)
                while len(top_pixel_counts) < MAX_K_TO_ANALYZE: top_pixel_counts.append(np.nan)
                while len(top_gabor_high_features) < MAX_K_TO_ANALYZE: top_gabor_high_features.append(np.nan)
                while len(top_gabor_low_features) < MAX_K_TO_ANALYZE: top_gabor_low_features.append(np.nan)
                
                row_data = [i]  + top_shap_probs + top_intensities + top_pixel_counts + top_gabor_high_features + top_gabor_low_features
                temp_results[predicted_class].append(row_data)
                
                labels_to_save = top_segment_label[:]
                while len(labels_to_save) < MAX_K_TO_ANALYZE:
                    labels_to_save.append(-1)
                segment_label_topk_nsamples[predicted_class].append([i] + labels_to_save)

                valid_images_count += 1
            
            except Exception as e:
                print(f"\n[ERROR] Lỗi khi xử lý ảnh index {i}: {e}")
                print(traceback.format_exc())
                continue # Bỏ qua ảnh lỗi và tiếp tục ảnh tiếp theo

        # --- GIAI ĐOẠN 2 & 3 & 4 (Giữ nguyên logic cũ, chỉ indent lại nếu cần) ---
        print(f"\nĐã xử lý thành công {valid_images_count} ảnh.")

        # Tính toán K_final
        if not k_votes:
            print("Không có vote nào hợp lệ. Đặt k_final = 3.")
            k_final = 3
        else:
            counts = Counter(k_votes)
            max_freq = max(counts.values())
            tied_ks = [k for k, freq in counts.items() if freq == max_freq]
            k_final = max(tied_ks)
            print(f"Final K is {k_final} based on votes: {counts}")

        if k_final > MAX_K_TO_ANALYZE:
            k_final = MAX_K_TO_ANALYZE

        # Sắp xếp và Cắt tỉa
        result_list = []
        num_full_cols = 1 + 5 * MAX_K_TO_ANALYZE
        for class_idx, matrix_list in sorted(temp_results.items()):
            if matrix_list:
                result_list.append((class_idx, np.array(matrix_list)))
            else:
                result_list.append((class_idx, np.empty((0, num_full_cols))))
        
        final_result_list = []
        cols_to_keep = [0] # ID
        cols_to_keep.extend(range(1, k_final + 1))
        cols_to_keep.extend(range(MAX_K_TO_ANALYZE + 1, MAX_K_TO_ANALYZE + k_final + 1))
        cols_to_keep.extend(range(2 * MAX_K_TO_ANALYZE + 1, 2 * MAX_K_TO_ANALYZE + k_final + 1))
        cols_to_keep.extend(range(3 * MAX_K_TO_ANALYZE + 1, 3 * MAX_K_TO_ANALYZE + k_final + 1))
        cols_to_keep.extend(range(4 * MAX_K_TO_ANALYZE + 1, 4 * MAX_K_TO_ANALYZE + k_final + 1))

        for class_idx, matrix_np in result_list:
            if matrix_np.size > 0:
                final_result_list.append((class_idx, matrix_np[:, cols_to_keep]))
            else:
                num_final_cols_trimmed = 1 + 5 * k_final
                final_result_list.append((class_idx, np.empty((0, num_final_cols_trimmed))))
        
        self.aggregated_intensities_tuple = tuple(final_result_list)
        
        # Tính Mean và Representative
        self.mean(k_final) 
        representative_samples = self.it_should_be_like_that()
        representative_masks = self.generate_binary_masks(representative_samples, segment_label_topk_nsamples, k_final)
        self.calculate_and_save_empirical_distances(k_final, 'empirical_distances.txt')

        # Lưu file
        if output_filename is None:
            output_filename = f"model_stats_{self.Ood_name or 'default'}.npz"
        elif not output_filename.endswith('.npz'):
            output_filename = output_filename.rsplit('.', 1)[0] + '.npz'
            
        print(f"Đang lưu kết quả tối ưu vào file: {output_filename}...")
        try:
            np.savez_compressed(
                output_filename, 
                k_final=k_final,
                mean_features=self.mean_features,
                inv_cov_matrices=self.inv_cov_matrices
            )
            print(f"Đã lưu file thành công: {output_filename}")
            return k_final, self.mean_features, self.inv_cov_matrices

        except Exception as e:
            print(f"ĐÃ XẢY RA LỖI trong quá trình lưu file .npz: {e}")
            traceback.print_exc()
            return k_final, self.mean_features, self.inv_cov_matrices
    # def extract_and_save_statistics(self, output_filename=None):
    #     """
    #     Thực hiện quy trình trích xuất đặc trưng đầy đủ (tương tự Extract)
    #     nhưng KHÔNG tính toán mẫu đại diện.
        
    #     Thay vào đó, nó sẽ lưu các mô hình thống kê quan trọng
    #     (k_final, mean_features, inv_cov_matrices) vào một file .txt.
    #     """
    #     print(f"\n--- Bắt đầu quy trình trích xuất và lưu thống kê ---")
        
    #     # Đặt tên file output mặc định nếu không được cung cấp
    #     if output_filename is None:
    #         output_filename = f"model_stats_{self.Ood_name or 'default'}.txt"

    #     # --- GIAI ĐOẠN 1: TRÍCH XUẤT (Tương tự hàm Extract) ---
    #     MAX_K_TO_ANALYZE = 10 # Cố định k tối đa = 10
    #     k_votes = [] # List để lưu phiếu bầu k_j của mỗi ảnh
    #     temp_results = {class_idx: [] for class_idx in range(self.num_classes)}
    #     segment_label_topk_nsamples = {class_idx: [] for class_idx in range(self.num_classes)}

    #     # 1. HÀM HELPER: Giải chuẩn hóa về numpy float [0, 1]
    #     def get_float_image_from_tensor(tensor_image):
    #         std = np.array(self.transform_std, dtype=np.float32)
    #         mean = np.array(self.transform_mean, dtype=np.float32)
    #         np_image = tensor_image.cpu().numpy().transpose((1, 2, 0))
    #         np_image = std * np_image + mean
    #         return np.clip(np_image, 0, 1)

    #     print(f"Bắt đầu xử lý {len(self.background_data)} ảnh nền...")
    #     for i, image_tensor in enumerate(tqdm(self.background_data, desc="Processing Images for Stats")):
    #         with torch.no_grad():
    #             logits = self.model(image_tensor.unsqueeze(0).to(self.device))
    #             predicted_class = torch.argmax(logits, dim=1).item()

    #         # 2. CHUẨN BỊ ẢNH VÀ PHÂN MẢNH
    #         image_numpy_float = get_float_image_from_tensor(image_tensor)
    #         segments_slic = slic(image_numpy_float, n_segments=self.n_segments,
    #                             compactness=self.compactness, sigma=self.sigma, start_label=self.start_label)
    #         num_actual_superpixels = len(np.unique(segments_slic))
    #         background_color = image_numpy_float.mean((0, 1))

    #         transform_for_prediction = transforms.Compose([
    #                 transforms.ToTensor(),
    #                 transforms.Resize((224, 224)),
    #                 transforms.Normalize(self.transform_mean, self.transform_std)
    #             ])
            
    #         # 3. PREDICTION_FUNCTION "VECTOR HÓA"
    #         def prediction_function(z):
    #             batch_size = 20
    #             all_logits = []
    #             h, w, c = image_numpy_float.shape
    #             unique_labels = np.unique(segments_slic)
    #             for i in range(0, z.shape[0], batch_size):
    #                 z_batch = z[i:i + batch_size]
    #                 masked_images_np = []
                    
    #                 for mask in z_batch:
    #                     temp_image = image_numpy_float.copy()
    #                     inactive_segments = np.where(mask == 0)[0]
    #                     inactive_labels = unique_labels[inactive_segments]
    #                     mask_all_inactive = np.isin(segments_slic, inactive_labels)
    #                     temp_image[mask_all_inactive] = background_color
    #                     masked_images_np.append(temp_image)
                    
    #                 tensors = torch.stack(
    #                     [transform_for_prediction(img) for img in masked_images_np]
    #                 ).to(self.device)

    #                 with torch.no_grad():
    #                     logits_shap = self.model(tensors)
    #                 all_logits.append(logits_shap.cpu().numpy())
    #             return np.concatenate(all_logits, axis=0)
            
    #         # 4. CHẠY SHAP
    #         explainer = KernelExplainer(prediction_function, np.zeros((1, num_actual_superpixels)))
    #         shap_values_for_image = explainer.shap_values(np.ones((1, num_actual_superpixels)), nsamples=2048 + num_actual_superpixels*2)

    #         # 5. TÍNH TOÁN K_J (Top-K cho ảnh)
    #         shap_values_for_predicted_class = shap_values_for_image[0, :, predicted_class]
    #         positive_shap_indices = np.where(shap_values_for_predicted_class > 0)[0]
            
    #         if len(positive_shap_indices) == 0:
    #             num_cols_to_pad = 5 * MAX_K_TO_ANALYZE
    #             row_data = [i] + [np.nan] * num_cols_to_pad
    #             temp_results[predicted_class].append(row_data)
    #             continue
            
    #         positive_shap_values = shap_values_for_predicted_class[positive_shap_indices]
    #         shap_label_pairs = list(zip(positive_shap_values, positive_shap_indices))
    #         sorted_by_shap = sorted(shap_label_pairs, key=lambda x: x[0], reverse=True)
    #         sorted_positive_values = [x[0] for x in sorted_by_shap]
            
    #         if len(sorted_positive_values) <= 1:
    #             k_votes.append(1)
    #         else:
    #             # k_gap
    #             k_gap = 1
    #             values_to_compare_gap = sorted_positive_values[:MAX_K_TO_ANALYZE]
    #             if len(values_to_compare_gap) > 1:
    #                 drops = [values_to_compare_gap[k] - values_to_compare_gap[k+1] for k in range(len(values_to_compare_gap) - 1)]
    #                 if drops:
    #                     k_j_index = np.argmax(drops)
    #                     k_gap = k_j_index + 1
                
    #             # k_constraint
    #             k_constraint = 1
    #             total_positive_shap = np.sum(sorted_positive_values)
    #             if total_positive_shap > 0:
    #                 cumulative_sum = 0
    #                 k_constraint_found = False
    #                 values_to_compare_constraint = sorted_positive_values[:MAX_K_TO_ANALYZE]
                    
    #                 for k_index in range(len(values_to_compare_constraint)):
    #                     cumulative_sum += values_to_compare_constraint[k_index]
    #                     if cumulative_sum > (0.5 * total_positive_shap):
    #                         k_constraint = k_index + 1
    #                         k_constraint_found = True
    #                         break
    #                 if not k_constraint_found:
    #                     k_constraint = len(values_to_compare_constraint) if values_to_compare_constraint else 1
                
    #             # Quyết định k_j
    #             k_j = max(k_gap, k_constraint)
    #             k_votes.append(k_j)

    #         # 6. TRÍCH XUẤT ĐẶC TRƯNG "DƯ THỪA" (CHO MAX_K)
    #         gabor_high, gabor_low = self.get_gabor_energy(image_numpy_float)
    #         segments_to_extract = sorted_by_shap[:MAX_K_TO_ANALYZE]
    #         top_intensities, top_shap_probs, top_pixel_counts, top_gabor_high_features, top_gabor_low_features, top_segment_label = [], [], [], [], [], []
            
    #         total_positive_shap = np.sum(positive_shap_values) # Tính lại tổng SHAP dương
            
    #         for shap_values, segment_label in segments_to_extract:
    #             prop = round(shap_values / total_positive_shap, 3) if total_positive_shap > 0 else 0
    #             top_shap_probs.append(prop)
    #             top_segment_label.append(segment_label)
                
    #             mask = (segments_slic == segment_label)
    #             pixels_in_segment = image_numpy_float[mask]
    #             pixel_count = pixels_in_segment.shape[0]
    #             top_pixel_counts.append(pixel_count)
    #             gabor_high_vals = gabor_high[mask]
    #             gabor_low_vals = gabor_low[mask]
    #             #Gabor
    #             if gabor_high_vals.size > 0:
    #                 top_gabor_high_features.append(np.mean(gabor_high_vals))
    #             else:
    #                 top_gabor_high_features.append(np.nan)
    #             if gabor_low_vals.size > 0:
    #                 top_gabor_low_features.append(np.mean(gabor_low_vals))
    #             else:
    #                 top_gabor_low_features.append(np.nan)
    #             if pixels_in_segment.size > 0:
    #                 avg_intensity = np.mean(pixels_in_segment) * 255
    #                 top_intensities.append(avg_intensity)
    #             else:
    #                 top_intensities.append(np.nan)

    #         # Làm đầy dữ liệu
    #         while len(top_intensities) < MAX_K_TO_ANALYZE: top_intensities.append(np.nan)
    #         while len(top_shap_probs) < MAX_K_TO_ANALYZE: top_shap_probs.append(np.nan)
    #         while len(top_pixel_counts) < MAX_K_TO_ANALYZE: top_pixel_counts.append(np.nan)
    #         while len(top_gabor_high_features) < MAX_K_TO_ANALYZE: top_gabor_high_features.append(np.nan)
    #         while len(top_gabor_low_features) < MAX_K_TO_ANALYZE: top_gabor_low_features.append(np.nan)
    #         row_data = [i]  + top_shap_probs + top_intensities + top_pixel_counts + top_gabor_high_features + top_gabor_low_features
    #         temp_results[predicted_class].append(row_data)
    #         labels_to_save = top_segment_label[:]
    #         while len(labels_to_save) < MAX_K_TO_ANALYZE:
    #             labels_to_save.append(-1) # Dùng -1 vì label SLIC luôn >= 0
    #         segment_label_topk_nsamples[predicted_class].append([i] + labels_to_save)
            
    #     # --- GIAI ĐOẠN 2: TÍNH TOÁN K_FINAL VÀ CẮT TỈA ---
    #     if not k_votes:
    #         print("Due to amount of votes are none so that K is 3.")
    #         k_final = 3
    #     else:
    #         counts = Counter(k_votes)
    #         max_freq = max(counts.values())
    #         tied_ks = [k for k, freq in counts.items() if freq == max_freq]
    #         k_final = max(tied_ks)
    #         print(f"Final K is {k_final} based on votes: {counts}")

    #     if k_final > MAX_K_TO_ANALYZE:
    #         print(f"!!! Cảnh báo: k_final ({k_final}) vượt quá MAX_K_TO_ANALYZE. Giới hạn k_final = {MAX_K_TO_ANALYZE}.")
    #         k_final = MAX_K_TO_ANALYZE

    #     # Sắp xếp và Cắt tỉa result_list (dữ liệu đặc trưng)
    #     result_list = []
    #     num_full_cols = 1 + 5 * MAX_K_TO_ANALYZE
    #     for class_idx, matrix_list in sorted(temp_results.items()):
    #         if matrix_list:
    #             result_list.append((class_idx, np.array(matrix_list)))
    #         else:
    #             result_list.append((class_idx, np.empty((0, num_full_cols))))
        
    #     final_result_list = []
    #     cols_to_keep = [0] # Cột ID
    #     cols_to_keep.extend(range(1, k_final + 1)) # Cột shap_prob
    #     cols_to_keep.extend(range(MAX_K_TO_ANALYZE + 1, MAX_K_TO_ANALYZE + k_final + 1)) # Cột intensity
    #     cols_to_keep.extend(range(2 * MAX_K_TO_ANALYZE + 1, 2 * MAX_K_TO_ANALYZE + k_final + 1)) # Cột pixel_count
    #     cols_to_keep.extend(range(3 * MAX_K_TO_ANALYZE + 1, 3 * MAX_K_TO_ANALYZE + k_final + 1)) # Cột pixel_count
    #     cols_to_keep.extend(range(4 * MAX_K_TO_ANALYZE + 1, 4 * MAX_K_TO_ANALYZE + k_final + 1)) # Cột pixel_count

    #     for class_idx, matrix_np in result_list:
    #         if matrix_np.size > 0:
    #             final_result_list.append((class_idx, matrix_np[:, cols_to_keep]))
    #         else:
    #             num_final_cols_trimmed = 1 + 5 * k_final
    #             final_result_list.append((class_idx, np.empty((0, num_final_cols_trimmed))))
        
    #     self.aggregated_intensities_tuple = tuple(final_result_list)
        
    #     # --- GIAI ĐOẠN 3: TÍNH TOÁN THỐNG KÊ ---
        
    #     # Gọi hàm mean để tính toán và gán self.mean_features, self.inv_cov_matrices
    #     self.mean(k_final) 
    #     representative_samples = self.it_should_be_like_that()
        
    #     # Biến này chính là cái ta cần lưu:
    #     representative_masks = self.generate_binary_masks(representative_samples, segment_label_topk_nsamples)
    #     self.calculate_and_save_empirical_distances(k_final, 'empirical_distances.txt')

    #     # --- GIAI ĐOẠN 4: LƯU FILE THEO YÊU CẦU ---
        
    #     print(f"Saving the file {output_filename}...")
        
    #     if output_filename is None:
    #         output_filename = f"model_stats_{self.Ood_name or 'default'}.npz"
    #     elif not output_filename.endswith('.npz'):
    #         output_filename = output_filename.rsplit('.', 1)[0] + '.npz'
            
    #     print(f"Đang lưu kết quả tối ưu vào file: {output_filename}...")

    #     try:
    #         # Sử dụng np.savez_compressed để lưu nhiều array vào 1 file
    #         # Nó nhanh, nén, và giữ nguyên kiểu dữ liệu/shape
    #         np.savez_compressed(
    #             output_filename, 
    #             k_final=k_final,  # Lưu k_final (số nguyên)
    #             mean_features=self.mean_features, # Lưu array 3D
    #             inv_cov_matrices=self.inv_cov_matrices            )
            
    #         print(f"Đã lưu file thành công: {output_filename}")
            
    #         # Trả về các giá trị đã tính, phòng trường hợp bạn muốn dùng
    #         return k_final, self.mean_features, self.inv_cov_matrices

    #     except Exception as e:
    #         print(f"ĐÃ XẢY RA LỖI trong quá trình lưu file .npz: {e}")
    #         import traceback
    #         traceback.print_exc()
    #         # Vẫn trả về giá trị dù lưu file lỗi
    #         return k_final, self.mean_features, self.inv_cov_matrices

    #                 # Vẫn trả về giá trị dù lưu file lỗi

 