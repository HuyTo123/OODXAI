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
                 n_segments = 50, compactness = 10, sigma = 1, start_label = 1, transform_mean=[0.485, 0.456, 0.406], transform_std=[0.229, 0.224, 0.225],
                 image_numpy_unnormalized = None, num_samples=100):
        """
        Subclass for KernelSHAP. __init__ chỉ dùng để lưu cấu hình.
        """
        # --- Super class init ---
        # `sample` ở đây là ảnh đã xử lý, dùng để tính OOD score
        super().__init__(model, Ood_name, background_data, sample, device, class_name)

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
        self.background_color = 0
        # --- State parameters ---
        # Khởi tạo các biến trạng thái, sẽ được điền giá trị sau
        self.segments_slic = None
        self.aggregated_intensities_tuple = ()
        self.uncertainty_segments= None
        self.calculate_Ood_scores()


        print("-> OodKernelExplainer đã được tạo và cấu hình. Sẵn sàng hoạt động.")
    
    def extract_segment_features(self, top_k=3):
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
            background_color=self.background_color
        )
        
        # 4. Gọi đúng phương thức `explain` và trả về kết quả
        self.aggregated_intensities_tuple = segment_analyzer.Extract(top_k=top_k)
        print(self.aggregated_intensities_tuple)
        print("--- Hoàn tất trích xuất ---")
        
    def explain(self):
        """
        Đây là phương thức CÔNG KHAI DUY NHẤT để chạy toàn bộ quy trình.
        Nó sẽ tự động làm mọi thứ: phân vùng, tạo ảnh, dự đoán và tính SHAP.
        """
        print("\n--- Bắt đầu quy trình giải thích của KernelSHAP ---")
        # Calculate OOD score for the sample
        # 1. Phân vùng ảnh bằng Superpixel
        self.segments_slic = slic(self.image_numpy_unnormalized, n_segments=self.n_segments,
                                  compactness=self.compactness, sigma=self.sigma, start_label=self.start_label)
        num_actual_superpixels = len(np.unique(self.segments_slic))
        print(f"1. Phân vùng ảnh thành {num_actual_superpixels} siêu pixel.")

        # 2. Định nghĩa hàm dự đoán nội bộ
        # Hàm này sẽ được truyền vào KernelExplainer và được gọi tự động
        transform_for_prediction = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.transform_mean, self.transform_std)
        ])

        def prediction_function(z):
            # `z` là một mảng các mặt nạ nhị phân do SHAP cung cấp
            masked_images_np = []
            for mask in z:
                temp_image = self.image_numpy_unnormalized.copy()
                # Tuple with 1 array in dim 0
                inactive_segments = np.where(mask == 0)[0]
                # Compare label of segments_slic with inactive_segments, transform segment  to True or False, then change to background color for True segments
                for seg_idx in inactive_segments:
                    temp_image[self.segments_slic == seg_idx] = self.background_color
                masked_images_np.append(temp_image)

            # Chuyển đổi hàng loạt ảnh sang tensor và dự đoán
            tensors = torch.stack(
                [transform_for_prediction(Image.fromarray(img.astype(np.uint8))) for img in masked_images_np]
            ).to(self.device)
            
            self.model.eval()
            with torch.no_grad():
                logits = self.model(tensors)
            return logits.cpu().numpy()

        # 3. Khởi tạo KernelExplainer và tính toán SHAP values
        print(f"2. Bắt đầu tính toán SHAP values với {self.num_samples} mẫu...")
        explainer = KernelExplainer(prediction_function, np.zeros((1, num_actual_superpixels)))
        self.shap_values = explainer.shap_values(np.ones((1, num_actual_superpixels)), nsamples=self.num_samples)
        print("3. Tính toán SHAP values hoàn tất!")
        # Tính toán các segment bị uncertainity
        self.uncertainty_segments = self.uncertainty ()
        print(self.uncertainty_segments)

        return self # Trả về self để có thể gọi .plot() nối tiếp
    def uncertainty (self):
        temp_results = {class_idx: [] for class_idx in range(self.num_classes)}

        for class_idx in range(self.num_classes):
            # Lấy SHAP values của class hiện tại
            shap_values_for_class = self.shap_values[0, :, class_idx]
            
            # 1. Lấy TẤT CẢ các segment có SHAP value > 0
            positive_shap_indices = np.where(shap_values_for_class > 0)[0]
            
            if len(positive_shap_indices) == 0:
                continue

            # 2. Tính toán cường độ sáng cho TỪNG segment và lưu dưới dạng tuple (cường độ, label)
            intensity_label_pairs = []
            for segment_label in positive_shap_indices:
                mask = (self.segments_slic == segment_label)
                pixels_in_segment = self.image_numpy_unnormalized[mask]
                
                # Bỏ qua nếu segment rỗng để tránh lỗi
                if pixels_in_segment.size == 0:
                    continue
                    
                avg_intensity = np.mean(pixels_in_segment)
                
                # Thêm tuple (cường độ, label) vào danh sách
                intensity_label_pairs.append((avg_intensity, int(segment_label))) # Chuyển label sang int cho an toàn

            # 3. Sắp xếp danh sách các tuple dựa trên cường độ sáng (phần tử đầu tiên)
            #    Hàm lambda x: x[0] chỉ định rằng việc sắp xếp sẽ dựa trên giá trị đầu tiên của mỗi tuple.
            sorted_pairs = sorted(intensity_label_pairs, key=lambda x: x[0], reverse= True)
            
            # 4. Lưu danh sách đã sắp xếp vào kết quả tạm thời
            temp_results[class_idx] = sorted_pairs[:5]

        # --- Phần chuyển đổi sang tuple cuối cùng vẫn giữ nguyên ---
        result_list = []
        for class_idx, values in sorted(temp_results.items()):
            result_list.append((class_idx, values))

        sample_uncertainty_tuple = tuple(result_list)
        print(sample_uncertainty_tuple)
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



    def plot(self, class_names=None):
        self.visualization.plot_kernelshap_with_uncertainty(self.image_numpy_unnormalized, 
                                           class_names=self.class_name, 
                                           segmentation=self.segments_slic, 
                                           shap_values=self.shap_values,
                                            unsafe_segments_tuple=self.uncertainty_segments,
                                           ood_percentile=self.ood_percentile,
                                           sample_scores=self.sample_scores,
                                           probs=self.probs,
                                           detector=self.Detector)
    