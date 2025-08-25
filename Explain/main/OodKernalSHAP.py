import numpy as np
from skimage.segmentation import slic
from .. import KernelExplainer
from tqdm.autonotebook import tqdm
import torch
# Giả sử OODExplainerBase đã được định nghĩa
from .OodXAIBase import OODExplainerBase
import torchvision.transforms as transforms
from PIL import Image

class OodKernelExplainer(OODExplainerBase):
    def __init__(self, model=None, Ood_name=None, background_data=None, sample=None, device=None, 
                 n_segments = 50, compactness = 30, sigma = 3, start_label = 1, transform_mean=[0.485, 0.456, 0.406], transform_std=[0.229, 0.224, 0.225],
                 image_numpy_unnormalized = None, num_samples=100):
        """
        Subclass for KernelSHAP. __init__ chỉ dùng để lưu cấu hình.
        """
        # --- Super class init ---
        # `sample` ở đây là ảnh đã xử lý, dùng để tính OOD score
        super().__init__(model, Ood_name, background_data, sample, device)

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
        self.background_color = 0
        # --- State parameters ---
        # Khởi tạo các biến trạng thái, sẽ được điền giá trị sau
        self.segments_slic = None
        self.calculate_Ood_scores()

        print("-> OodKernelExplainer đã được tạo và cấu hình. Sẵn sàng hoạt động.")

    def explain(self):
        """
        Đây là phương thức CÔNG KHAI DUY NHẤT để chạy toàn bộ quy trình.
        Nó sẽ tự động làm mọi thứ: phân vùng, tạo ảnh, dự đoán và tính SHAP.
        """
        print("\n--- Bắt đầu quy trình giải thích của KernelSHAP ---")
        # Calculate OOD score for the sample
        # 1. Phân vùng ảnh bằng Superpixel
        self.segments_slic = slic(self.image_numpy_unnormalized, n_segments=self.n_segments,
                                  compactness=10, sigma=1, start_label=1)
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
                inactive_segments = np.where(mask == 0)[0]
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
        return self # Trả về self để có thể gọi .plot() nối tiếp

    def plot(self, class_names=None):
        self.visualization.plot_kernelshap(self.image_numpy_unnormalized, 
                                           class_names=class_names, 
                                           segmentation=self.segments_slic, 
                                           shap_values=self.shap_values,
                                           ood_percentile=self.ood_percentile,
                                           sample_scores=self.sample_scores,
                                           probs=self.probs,
                                           detector=self.Detector)
    