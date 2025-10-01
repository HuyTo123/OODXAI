import os, sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import torchvision
import numpy as np
import cv2
import shap
from PIL import Image
from skimage.segmentation import slic
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
import shap

# Thêm đường dẫn project vào sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Giả sử package của bạn đã được cài đặt và có thể import
from oodxai import OodKernelExplainer
# ==============================================================================
# 1. KHAI BÁO VÀ THIẾT LẬP
# ==============================================================================
print("--- BẮT ĐẦU SCRIPT ---")

MODEL_PATH = 'testmodel/cat_dogs_huggingface/cat_dog_resnet50_best.pth'
IMAGE_PATH = "testmodel/cat_dogs_huggingface/CatandDog_segment_noise/Cat/Cat_1.png"
DATA_ROOT = 'testmodel/cat_dogs_huggingface/Cat_and_Dog_Images/test'

NUM_SUPERPIXELS = 50
NUM_SAMPLES = 500
TRANSFORM_MEAN = [0.485, 0.456, 0.406]
TRANSFORM_STD = [0.229, 0.224, 0.225]

CLASS_NAMES = ['Cat', 'Dog']

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Sử dụng thiết bị: {DEVICE}")

num_runs = 50

# ==============================================================================
# 2. HÀM TẢI MODEL
# ==============================================================================
def load_model(num_classes, device):
    model = models.resnet50(weights=None) # Chỉ cần kiến trúc
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    return model # <<<--- SỬA LỖI 1: Thêm return model



# ==============================================================================
# 4. THỰC THI CHÍNH
# ==============================================================================
if __name__ == '__main__':
    # --- Các bước tải model và chuẩn bị ảnh giữ nguyên ---
    if not os.path.exists(MODEL_PATH):
        print(f"LỖI: Không tìm thấy model tại: {MODEL_PATH}")
        exit()
    model = load_model(num_classes=2, device=DEVICE)
    transform_for_slic = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    image = cv2.imread(IMAGE_PATH)
    if image is None:
        print(f"LỖI: Không đọc được ảnh tại: {IMAGE_PATH}")
        exit()
        
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)

    image_tensor_unnormalized = transform_for_slic(pil_image)
    image_numpy_unnormalized = image_tensor_unnormalized.permute(1, 2, 0).numpy()
    segments_slic = slic(image_numpy_unnormalized, n_segments=NUM_SUPERPIXELS,
                         compactness=10, sigma=1, start_label=0)
    num_actual_superpixels = len(np.unique(segments_slic))
    print(f"1. Phân vùng ảnh thành {num_actual_superpixels} siêu pixel.")

    transform_for_prediction = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(TRANSFORM_MEAN, TRANSFORM_STD)
    ])
    
    background_color = image_numpy_unnormalized.mean((0, 1))

    # prediction_function giữ nguyên, tôi rút gọn để dễ đọc
    def prediction_function(z):
        # (Nội dung hàm prediction_function của bạn ở đây)
        batch_size = 10 
        all_logits = []
        unique_labels = np.unique(segments_slic)

        for i in range(0, z.shape[0], batch_size):
            z_batch = z[i:i + batch_size]
            masked_images_np = []
            for mask in z_batch:
                temp_image = image_numpy_unnormalized.copy()
                inactive_segments = np.where(mask == 0)[0]
                inactive_labels = unique_labels[inactive_segments]
                mask_all_inactive = np.isin(segments_slic, inactive_labels)
                temp_image[mask_all_inactive] = background_color
                masked_images_np.append(temp_image)
            tensors = torch.stack([transform_for_prediction(img) for img in masked_images_np]).to(DEVICE)
            with torch.no_grad():
                logits = model(tensors)
            all_logits.append(logits.cpu().numpy())
        return np.concatenate(all_logits, axis=0)

    # --- BẮT ĐẦU PHẦN THỐNG KÊ ---
    print(f"\nBắt đầu chạy {num_runs} lần KernelSHAP để lấy số liệu thống kê...")

    # 1. Khởi tạo các biến lưu trữ kết quả
    # Đếm số lần mỗi segment xuất hiện là tích cực/tiêu cực
    positive_counts = np.zeros(num_actual_superpixels)
    negative_counts = np.zeros(num_actual_superpixels)
    # Tính tổng SHAP values để sau đó lấy trung bình
    shap_value_sums = np.zeros((num_actual_superpixels, 2)) # 2 là num_classes

    # 2. Xác định lớp dự đoán một lần duy nhất bên ngoài vòng lặp
    with torch.no_grad():
        logits = model(transform_for_prediction(pil_image).unsqueeze(0).to(DEVICE))
        predicted_class = torch.argmax(logits, dim=1).item()
        predicted_class_name = CLASS_NAMES[predicted_class]
        predicted_score = logits[0, predicted_class].item()
    
    # 3. Khởi tạo explainer một lần
    explainer = shap.KernelExplainer(prediction_function, np.zeros((1, num_actual_superpixels)))

    # 4. Vòng lặp chính để chạy 100 lần
    for _ in tqdm(range(num_runs), desc="Chạy thống kê SHAP"):
        # Tính SHAP values cho lần chạy hiện tại
        shap_values = explainer.shap_values(np.ones((1, num_actual_superpixels)), nsamples=NUM_SAMPLES)
        
        # Giả định shap_values là mảng 3D (ảnh, segment, class)
        shap_values_for_this_run = shap_values[0, :, :] # Lấy dữ liệu cho ảnh duy nhất
        
        # Lấy SHAP values cho lớp đã dự đoán
        shap_values_for_predicted_class = shap_values_for_this_run[:, predicted_class]

        # Tìm các segment có giá trị dương và âm
        positive_indices = np.where(shap_values_for_predicted_class > 0)[0]
        negative_indices = np.where(shap_values_for_predicted_class < 0)[0]
        
        # Cập nhật bộ đếm
        positive_counts[positive_indices] += 1
        negative_counts[negative_indices] += 1
        
        # Cộng dồn SHAP values vào tổng
        shap_value_sums += shap_values_for_this_run

    # --- KẾT THÚC VÒNG LẶP THỐNG KÊ ---

    # 5. Tính toán kết quả cuối cùng
    positive_prob = positive_counts / num_runs
    negative_prob = negative_counts / num_runs
    mean_shap_values = shap_value_sums / num_runs
    shap_values_for_pred_class = mean_shap_values[:, predicted_class]
    # np.argsort return incices sắp xếp tăng dần, đảo ngược để lấy top 5 lớn nhất
    sorted_indices = np.argsort(shap_values_for_pred_class)[::-1]
    top_5_segment_ids = sorted_indices[:5]





    # 6. In kết quả ra màn hình
    print("\n" + "="*80)
    print(f"KẾT QUẢ THỐNG KÊ SAU {num_runs} LẦN CHẠY (cho lớp dự đoán: {predicted_class})")
    print("="*80)
    print(f"{'Segment ID':<12} | {'P(Tích cực)':<15} | {'P(Tiêu cực)':<15} | {'Mean SHAP (Class 0)':<20} | {'Mean SHAP (Class 1)':<20}")
    print("-"*80)
    
    for i in range(num_actual_superpixels):
        print(f"{i:<12} | {positive_prob[i]:<15.2%} | {negative_prob[i]:<15.2%} | {mean_shap_values[i, 0]:<20.4f} | {mean_shap_values[i, 1]:<20.4f}")
    
    print("="*80)

    # 7
    # ==============================================================================
    # 7. LƯU KẾT QUẢ RA FILE TXT
    # ==============================================================================

    # 7.1. Tạo đường dẫn và tên file output
    # <<< THAY ĐỔI: Đặt đường dẫn lưu file cố định theo yêu cầu >>>
    # ==============================================================================
    # 7. LƯU KẾT QUẢ RA FILE TXT
    # ==============================================================================

    # 7.1. Tạo đường dẫn và tên file output
    output_dir = "testmodel/cat_dogs_huggingface/CatandDog_segment_with_noise"

    class_folder = os.path.basename(os.path.dirname(IMAGE_PATH))
    file_name_without_ext = os.path.splitext(os.path.basename(IMAGE_PATH))[0] +'top_k_noise'

    file_name_final = file_name_without_ext + '_final'

    final_output_dir = os.path.join(output_dir, class_folder)
    os.makedirs(final_output_dir, exist_ok=True)

    output_file_path_full = os.path.join(final_output_dir, f"{file_name_without_ext}.txt")
    output_file_path_final = os.path.join(final_output_dir, f"{file_name_final}.txt")

    # 7.2. Ghi nội dung đầy đủ vào file thứ nhất
    print(f"\nĐang lưu kết quả đầy đủ vào: {output_file_path_full}")
    try:
        with open(output_file_path_full, 'w', encoding='utf-8') as f_full:
            f_full.write("="*80 + "\n")
            f_full.write(f"KẾT QUẢ THỐNG KÊ SHAP\n")
            f_full.write(f"File ảnh gốc: {IMAGE_PATH}\n")

            # <<< THAY ĐỔI 1: Hiển thị chỉ số lớp và tất cả logit scores >>>
            all_scores = logits.cpu().numpy()[0]
            scores_str = ", ".join([f"{s:.4f}" for s in all_scores])
            f_full.write(f"Dự đoán: class {predicted_class} - Logit Scores: [{scores_str}]\n")
            # <<< KẾT THÚC THAY ĐỔI 1 >>>

            f_full.write(f"Segment Quan trọng - {', '.join(map(str, top_5_segment_ids))}\n")
            f_full.write("="*80 + "\n")
            f_full.write(f"{'Segment ID':<12} | {'P(Tích cực)':<15} | {'P(Tiêu cực)':<15} | {'Mean SHAP (Class 0)':<20} | {'Mean SHAP (Class 1)':<20}\n")
            f_full.write("-"*80 + "\n")
            
            for i in range(num_actual_superpixels):
                line = f"{i:<12} | {positive_prob[i]:<15.2%} | {negative_prob[i]:<15.2%} | {mean_shap_values[i, 0]:<20.4f} | {mean_shap_values[i, 1]:<20.4f}\n"
                f_full.write(line)
            
            f_full.write("="*80 + "\n")
        print("Lưu file đầy đủ thành công!")
    except Exception as e:
        print(f"LỖI: Không thể lưu file đầy đủ. Lỗi: {e}")

    # Ghi nội dung rút gọn (chỉ dữ liệu) vào file thứ hai
    print(f"\nĐang lưu kết quả rút gọn vào: {output_file_path_final}")
    try:
        with open(output_file_path_final, 'w', encoding='utf-8') as f_final:
            # <<< THAY ĐỔI 2: Chỉ lặp qua top 5 segment quan trọng nhất >>>
            for i in top_5_segment_ids:
                line = f"{i} {positive_prob[i]:.4f} {negative_prob[i]:.4f} {mean_shap_values[i, 0]:.4f} {mean_shap_values[i, 1]:.4f}\n"
                f_final.write(line)
            # <<< KẾT THÚC THAY ĐỔI 2 >>>
        print("Lưu file rút gọn thành công!")
    except Exception as e:
        print(f"LỖI: Không thể lưu file rút gọn. Lỗi: {e}")
