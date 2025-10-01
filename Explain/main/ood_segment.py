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

from sklearn.metrics.pairwise import cosine_similarity

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

    def Extract(self, top_k=3):
        """
        Trích xuất đặc trưng (phiên bản mở rộng).
        """
        print(f"\n--- Bắt đầu quy trình trích xuất đặc trưng (phiên bản 4 loại thông tin) ---")
        
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
                num_cols_to_pad = 4 * top_k
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
            top_k_segments = sorted_by_shap[:top_k]

            #  Chuẩn bị các bước tính toán tiếp theo
            top_intensities = []
            top_shap_probs = []
            top_pixel_counts = []
            top_segment_label = []
            
            for shap_values, segment_label in top_k_segments:
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
                if pixels_in_segment.size > 0:
                    avg_intensity = np.mean(pixels_in_segment) * 255
                    top_intensities.append(avg_intensity)
                else:
                    top_intensities.append(np.nan)
            # Làm đầy dữ liệu
            while len(top_intensities) < top_k:
                top_intensities.append(np.nan)
            while len(top_shap_probs) < top_k:
                top_shap_probs.append(np.nan)
            while len(top_pixel_counts) < top_k:
                top_pixel_counts.append(np.nan)
            while len(top_segment_label) < top_k:
                top_segment_label.append(np.nan)

            # Kết hợp dữ liệu thành một hàng
            row_data = [i]  + top_shap_probs + top_intensities + top_pixel_counts 
            row_segment = [i] + top_segment_label
            # Lưu kết quả vào đúng vị trí
            temp_results[predicted_class].append(row_data)
            segment_label_topk_nsamples[predicted_class].append(row_segment)
           

        result_list = []
        num_final_cols = 1 + 3 * top_k
        for class_idx, matrix_list in sorted(temp_results.items()):
            if matrix_list:
                matrix_np = np.array(matrix_list )
                result_list.append((class_idx, matrix_np))
            else:
                result_list.append((class_idx, np.empty((0, num_final_cols))))
        self.aggregated_intensities_tuple = tuple(result_list)
        self.mean(top_k)
        representative_samples = self.it_should_be_like_that()
        representative_segment_labels = self.get_labels_for_representatives(representative_samples, segment_label_topk_nsamples)
        print(f"\n--- Hoàn tất trích xuất đặc trưng. Kết quả có dạng ma trận mở rộng. ---")
        return self.aggregated_intensities_tuple, representative_segment_labels, self.mean_features, self.inv_cov_matrices
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
        print(f"\n--- Bắt đầu tính toán {top_k} mô hình thống kê cho từng bậc segment ---")
        
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
                    
                    # Lấy mean (sử dụng nanmean là rất tốt)
                    # Nó tự động xử lí NaN mà không cần phải thay thế 
                    mean_shap_prob = np.nanmean(shap_prob_col)
                    mean_intensity = np.nanmean(intensity_col)
                    mean_pixel_count = np.nanmean(pixel_count_col)
                    
                    # Nối dài danh sách đặc trưng cho lớp hiện tại
                    class_features.append([mean_shap_prob, mean_intensity, mean_pixel_count])

                    # --- BỔ SUNG: Tính toán ma trận hiệp phương sai cho k hiện tại ---
                    # 1. Ghép 3 cột đặc trưng lại thành một ma trận (số_ảnh, 3)
                    feature_matrix_k = np.stack([shap_prob_col, intensity_col, pixel_count_col], axis=1)

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
                    cov_matrix_k = np.cov(feature_matrix_k, rowvar=False) + np.identity(3) * 1e-6

                    # Ma trận thực tế rất có thể bị suy biến, nên ta dùng pseudo-inverse
                    # Ma trận suy biến DET = 0 
                    inv_cov_matrix_k = pinv(cov_matrix_k)
                    
                    # 4. Thêm ma trận vừa tính vào list của lớp hiện tại
                    class_inv_cov_list.append(inv_cov_matrix_k)

            else:
                class_features = np.full((top_k, 3), np.nan).tolist()
                # --- BỔ SUNG: Điền ma trận rỗng nếu không đủ dữ liệu ---
                class_inv_cov_list = np.full((top_k, 3, 3), np.nan)
                
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