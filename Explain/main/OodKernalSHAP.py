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
        self.aggregated_intensities_tuple = segment_analyzer.Extract()
        torch.cuda.empty_cache()

        print(len(self.aggregated_intensities_tuple[0][1]),len(self.aggregated_intensities_tuple[1][1]))
        print("--- Hoàn tất trích xuất ---")
        
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
        print(np.sum(shap_values_for_class2) + explainer.expected_value[1], 'ket qua', self.logits)
        print(np.sum(shap_values_for_class1) + explainer.expected_value[0], 'ket qua', self.logits)       

        print("3. Tính toán SHAP values hoàn tất!")
        # Tính toán các segment bị uncertainity
        self.uncertainty_segments = self.uncertainty ()

        return self # Trả về self để có thể gọi .plot() nối tiếp
    def uncertainty (self):
        temp_results = {class_idx: [] for class_idx in range(self.num_classes)}

        for class_idx in range(self.num_classes):
            # Lấy SHAP values của class hiện tại
            shap_values_for_class = self.shap_values[0, :, class_idx]       
            # 1. Lấy cả GIÁ TRỊ và VỊ TRÍ (label) của các segment có SHAP value > 0
            positive_shap_indices = np.where(shap_values_for_class > 0)[0]
            
            if len(positive_shap_indices) == 0:
                continue

            # Lấy các giá trị SHAP tương ứng
            positive_shap_values = shap_values_for_class[positive_shap_indices]
            
            # 2. Ghép cặp (SHAP value, label) lại với nhau
            shap_label_pairs = list(zip(positive_shap_values, positive_shap_indices))
            
            # 3. SẮP XẾP danh sách các cặp này dựa trên SHAP value (phần tử đầu tiên)
            #    Đây là thay đổi cốt lõi theo yêu cầu của bạn.
            sorted_by_shap = sorted(shap_label_pairs, key=lambda x: x[0], reverse=True)
            # 4. Bây giờ, tạo danh sách (cường độ, label) cuối cùng DỰA TRÊN THỨ TỰ ĐÃ SẮP XẾP MỚI
            final_pairs_for_class = []
            for shap_value, segment_label in sorted_by_shap:
                mask = (self.segments_slic == segment_label)
                pixels_in_segment = self.image_numpy_0_1[mask]
                
                if pixels_in_segment.size == 0:
                    continue
                    
                avg_intensity = np.mean(pixels_in_segment)
                
                # Thêm tuple (cường độ, label) vào danh sách cuối cùng.
                # Thứ tự của các phần tử trong danh sách này giờ đây là theo SHAP value giảm dần.
                final_pairs_for_class.append((avg_intensity, int(segment_label)))

            # 5. Lưu danh sách đã sắp xếp vào kết quả tạm thời
            temp_results[class_idx] = final_pairs_for_class[:3]

        # --- Phần chuyển đổi sang tuple cuối cùng vẫn giữ nguyên ---
        result_list = []
        for class_idx, values in sorted(temp_results.items()):
            result_list.append((class_idx, values))
        sample_uncertainty_tuple = tuple(result_list)
        unsafe_segments = {f"class_{i}": [] for i in range(self.num_classes)}

        # Lặp qua từng class để tính toán và so sánh
        for class_idx in range(self.num_classes):
            # 1. Lấy và xử lý dữ liệu nền cho class hiện tại
            background_list_original = self.aggregated_intensities_tuple[class_idx][1]
            background_list_cleaned = [item for item in background_list_original if not math.isnan(item)]
            
            # Bỏ qua nếu không có đủ dữ liệu nền để tính ngưỡng
            if len(background_list_cleaned) < 20: # Cần ít nhất 20 điểm dữ liệu
                print(f"Cảnh báo: Class {class_idx} không có đủ dữ liệu nền để xác định ngưỡng an toàn.")
                continue
                
            sorted_background_list = sorted(background_list_cleaned)

            # 2. Xác định ngưỡng an toàn cho class hiện tại
            list_len = len(sorted_background_list)
            lower_index = int(list_len * 0.1)
            upper_index = int(list_len * 0.9) - 1
            
            lower_bound = sorted_background_list[lower_index]
            upper_bound = sorted_background_list[upper_index]
            
            print(f"Ngưỡng an toàn cho Class {class_idx}: ({lower_bound:.2f}, {upper_bound:.2f})")

            # 3. So sánh các segment của sample với ngưỡng
            sample_pairs_for_class = sample_uncertainty_tuple[class_idx][1]
            for intensity, label in sample_pairs_for_class:
                if not (lower_bound < intensity < upper_bound):
                    # Gán vào dictionary kết quả theo đúng class_idx
                    key_name = f"class_{class_idx}"
                    unsafe_segments[key_name].append(label)
                    print(f"-> Class {class_idx}: Segment {label} (giá trị {intensity:.2f}) được xác định là KHÔNG an toàn.")
        
        # Trả về kết quả cuối cùng
        final_result_list = []
        # Sắp xếp dictionary theo key để đảm bảo thứ tự class (0, 1, 2...)
        for key, values in sorted(unsafe_segments.items()):
            # Tách chỉ số class (là số nguyên) từ key (là chuỗi 'class_0')
            class_index = int(key.split('_')[1])
            final_result_list.append((class_index, values))

        # Trả về kết quả cuối cùng dưới dạng một tuple lớn
        return tuple(final_result_list)

    def it_should_be_like_this(self):
        pass

    def plot(self, class_names=None):
        sample_s, percentile = self.calculate_Ood_scores()
        self.visualization.plot_kernelshap_with_uncertainty(self.image_numpy_0_1, 
                                           class_names=self.class_name, 
                                           segmentation=self.segments_slic, 
                                           shap_values=self.shap_values,
                                            unsafe_segments_tuple=self.uncertainty_segments,
                                           ood_percentile=percentile,
                                           sample_scores=sample_s,
                                           probs=self.probs,
                                           detector=self.Detector)
    