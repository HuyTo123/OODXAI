import os
import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
import cv2
import shap
from PIL import Image
from skimage.segmentation import slic
from tqdm.autonotebook import tqdm
from pathlib import Path

# ==============================================================================
# 1. KHAI BÁO VÀ CẤU HÌNH TRUNG TÂM
# ==============================================================================
print("--- KHỞI TẠO CẤU HÌNH ---")

# --- CẤU HÌNH VÒNG LẶP (Người dùng tùy chỉnh) ---
START_INDEX = 50     # Bắt đầu từ Cat_50
END_INDEX = 52      # Kết thúc ở Cat_100
BASE_IMAGE_DIR = "testmodel/cat_dogs_huggingface/Cat_and_Dog_Images/test/Cat/"
PREFIX_NAME = "Cat_" # Tiền tố tên file
EXTENSION = ".png"   # Đuôi file

# --- Cấu hình tham số khác ---
TOP_K = 3
MODEL_PATH = 'testmodel/cat_dogs_huggingface/cat_dog_resnet50_best.pth'
NUM_SUPERPIXELS = 50
NUM_SAMPLES = 500 
NUM_RUNS = 50 

TRANSFORM_MEAN = [0.485, 0.456, 0.406]
TRANSFORM_STD = [0.229, 0.224, 0.225]
CLASS_NAMES = ['Cat', 'Dog']

# Thư mục output
OUTPUT_DIR_ORIGINAL = "testmodel/cat_dogs_huggingface/CatandDog_segment"
OUTPUT_DIR_NOISE_IMG = "testmodel/cat_dogs_huggingface/CatandDog_segment_noise"
OUTPUT_DIR_NOISE_ANALYSIS = "testmodel/cat_dogs_huggingface/CatandDog_segment_with_noise"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Sử dụng thiết bị: {DEVICE}")


# ==============================================================================
# 2. CÁC HÀM TIỆN ÍCH
# ==============================================================================

def load_model(num_classes, device):
    """Tải kiến trúc model ResNet50 và load trọng số đã huấn luyện."""
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    if not os.path.exists(MODEL_PATH):
        print(f"LỖI: Không tìm thấy model tại: {MODEL_PATH}")
        sys.exit()
        
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def run_shap_analysis(model, current_image_path, top_k, output_base_dir, analysis_type='original'):
    """
    Hàm lõi: thực hiện phân tích SHAP trên một ảnh, lưu kết quả và trả về các thông tin cần thiết.
    """
    # --- A. Chuẩn bị ảnh và phân đoạn Superpixel ---
    transform_for_slic = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    
    image = cv2.imread(current_image_path)
    if image is None:
        print(f"LỖI: Không đọc được ảnh tại: {current_image_path}")
        return None, None, None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)

    image_tensor_unnormalized = transform_for_slic(pil_image)
    image_numpy_unnormalized = image_tensor_unnormalized.permute(1, 2, 0).numpy()
    segments_slic = slic(image_numpy_unnormalized, n_segments=NUM_SUPERPIXELS,
                         compactness=10, sigma=1, start_label=0)
    num_actual_superpixels = len(np.unique(segments_slic))
    print(f"Phân vùng ảnh thành {num_actual_superpixels} siêu pixel.")

    # --- B. Định nghĩa hàm dự đoán cho SHAP ---
    transform_for_prediction = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(TRANSFORM_MEAN, TRANSFORM_STD)
    ])
    background_color = image_numpy_unnormalized.mean((0, 1))

    def prediction_function(z):
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

    # --- C. Chạy vòng lặp thống kê SHAP ---
    print(f"Bắt đầu chạy {NUM_RUNS} lần KernelSHAP để lấy số liệu thống kê...")
    positive_counts = np.zeros(num_actual_superpixels)
    negative_counts = np.zeros(num_actual_superpixels)
    shap_value_sums = np.zeros((num_actual_superpixels, len(CLASS_NAMES)))

    with torch.no_grad():
        logits = model(transform_for_prediction(pil_image).unsqueeze(0).to(DEVICE))
        predicted_class = torch.argmax(logits, dim=1).item()
    
    explainer = shap.KernelExplainer(prediction_function, np.zeros((1, num_actual_superpixels)))

    # Dùng tqdm với tham số leave=False để không in quá nhiều dòng khi chạy nhiều ảnh
    for _ in tqdm(range(NUM_RUNS), desc="Chạy thống kê SHAP", leave=False):
        shap_values = explainer.shap_values(np.ones((1, num_actual_superpixels)), nsamples=NUM_SAMPLES, silent=True)
        shap_values_for_this_run = np.array(shap_values)[0, :, :]
        shap_values_for_predicted_class = shap_values_for_this_run[:, predicted_class]

        # Tìm các segment có giá trị dương và âm
        positive_indices = np.where(shap_values_for_predicted_class > 0)[0]
        negative_indices = np.where(shap_values_for_predicted_class < 0)[0]
        
        # Cập nhật bộ đếm
        positive_counts[positive_indices] += 1
        negative_counts[negative_indices] += 1
        
        # Cộng dồn SHAP values vào tổng
        shap_value_sums += shap_values_for_this_run

    # --- D. Tính toán và xác định Top K ---
    positive_prob = positive_counts / NUM_RUNS
    negative_prob = negative_counts / NUM_RUNS
    mean_shap_values = shap_value_sums / NUM_RUNS
    
    shap_values_for_pred_class = mean_shap_values[:, predicted_class]
    sorted_indices = np.argsort(shap_values_for_pred_class)[::-1]
    top_k_segment_ids = sorted_indices[:top_k]

    # --- E. Lưu kết quả ra file text (GIỮ NGUYÊN FORMAT CỦA BẠN) ---
    class_folder = os.path.basename(os.path.dirname(current_image_path))
    file_name_without_ext = Path(current_image_path).stem
    if analysis_type == 'noise':
        file_name_without_ext += '_top_k_noise'
    final_output_dir = os.path.join(output_base_dir, class_folder)
    os.makedirs(final_output_dir, exist_ok=True)
    
    output_file_path_full = os.path.join(final_output_dir, f"{file_name_without_ext}.txt")
    output_file_path_final = os.path.join(final_output_dir, f"{file_name_without_ext}_final.txt")

    print(f"\nĐang lưu kết quả vào thư mục: {final_output_dir}")
    
    # === KHÔI PHỤC FORMAT GHI FILE CỦA BẠN ===
    with open(output_file_path_full, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"KẾT QUẢ THỐNG KÊ SHAP\n")
        f.write(f"File ảnh: {current_image_path}\n")
        all_scores = logits.cpu().numpy()[0]
        scores_str = ", ".join([f"{s:.4f}" for s in all_scores])
        f.write(f"Dự đoán: class {predicted_class} - Logit Scores: [{scores_str}]\n")
        f.write(f"Top {top_k} Segment quan trọng: {', '.join(map(str, top_k_segment_ids))}\n")
        f.write("="*80 + "\n")
        f.write(f"{'Segment ID':<12} | {'P(Tích cực)':<15} | {'P(Tiêu cực)':<15} | {'Mean SHAP (Class 0)':<20} | {'Mean SHAP (Class 1)':<20}\n")
        f.write("-"*80 + "\n")
        for i in range(num_actual_superpixels):
            f.write(f"{i:<12} | {positive_prob[i]:<15.2%} | {negative_prob[i]:<15.2%} | {mean_shap_values[i, 0]:<20.4f} | {mean_shap_values[i, 1]:<20.4f}\n")
    
    with open(output_file_path_final, 'w', encoding='utf-8') as f:
        for i in sorted_indices[:5]:
            f.write(f"{i} {positive_prob[i]:.4f} {negative_prob[i]:.4f} {mean_shap_values[i, 0]:.4f} {mean_shap_values[i, 1]:.4f}\n")
    # ============================================

    print("Lưu file thành công!")
    return top_k_segment_ids, image_numpy_unnormalized, segments_slic


def create_and_save_noised_image(original_image_numpy, original_segments, labels_to_black, original_image_path, output_dir):
    """Tạo ảnh nhiễu bằng cách bôi đen các superpixel được chỉ định và lưu lại."""
    
    # --- Tạo bản sao ảnh để chỉnh sửa ---
    output_image = original_image_numpy.copy()

    # --- Lặp qua danh sách ID và bôi đen ---
    for label in labels_to_black:
        mask = original_segments == label
        output_image[mask] = [0, 0, 0] # Màu đen

    # --- Xây dựng đường dẫn và lưu file ---
    input_path = Path(original_image_path)
    class_name = input_path.parts[-2]
    filename = input_path.name
    
    output_class_dir = Path(output_dir) / class_name
    output_class_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_class_dir / filename
    
    # Chuyển đổi về định dạng ảnh 8-bit và lưu
    image_to_save_uint8 = (output_image * 255).astype(np.uint8)
    pil_image_to_save = Image.fromarray(image_to_save_uint8)
    pil_image_to_save.save(output_path)
    
    print(f"Ảnh nhiễu đã được lưu tại: {output_path}")
    return str(output_path)


def process_single_image(model, image_path, image_index):
    """Hàm điều phối quy trình cho 1 ảnh"""
    print(f"\n" + "#"*60)
    print(f" ĐANG XỬ LÝ ẢNH THỨ {image_index}: {os.path.basename(image_path)}")
    print(f"#"*60)

    # --- Bước 1: Phân tích ảnh gốc ---
    print(f"-> [Ảnh {image_index}] Bước 1: Phân tích ảnh gốc")
    top_k_ids, original_numpy, original_segments = run_shap_analysis(
        model, image_path, TOP_K, OUTPUT_DIR_ORIGINAL
    )
    
    if top_k_ids is None:
        print(f"-> [Ảnh {image_index}] BỎ QUA: Lỗi khi phân tích ảnh gốc.")
        return

    # --- Bước 2: Tạo ảnh nhiễu ---
    print(f"-> [Ảnh {image_index}] Bước 2: Tạo ảnh nhiễu")
    noised_image_path = create_and_save_noised_image(
        original_numpy, original_segments, top_k_ids, 
        image_path, OUTPUT_DIR_NOISE_IMG
    )

    # --- Bước 3: Phân tích ảnh nhiễu ---
    print(f"-> [Ảnh {image_index}] Bước 3: Phân tích ảnh nhiễu")
    
    # Tách thư mục output cho phân tích nhiễu để tránh đè file
    noise_file_name_stem = Path(noised_image_path).stem
    
    run_shap_analysis(
        model, noised_image_path, TOP_K, 
        OUTPUT_DIR_NOISE_ANALYSIS,
        analysis_type='noise'
    )
    print(f"-> [Ảnh {image_index}] HOÀN THÀNH.")


# ==============================================================================
# 3. LUỒNG THỰC THI CHÍNH (ĐÃ CẬP NHẬT VÒNG LẶP)
# ==============================================================================
if __name__ == '__main__':
    # --- Bước 0: Tải model một lần duy nhất ---
    print("Đang tải model...")
    model = load_model(num_classes=len(CLASS_NAMES), device=DEVICE)

    # --- Vòng lặp xử lý ---
    print(f"Bắt đầu vòng lặp từ {START_INDEX} đến {END_INDEX}...")
    
    for i in range(START_INDEX, END_INDEX + 1):
        # Tạo đường dẫn đầy đủ: .../Cat/Cat_50.png
        file_name = f"{PREFIX_NAME}{i}{EXTENSION}"
        current_img_path = os.path.join(BASE_IMAGE_DIR, file_name)

        # Kiểm tra xem file có tồn tại không
        if os.path.exists(current_img_path):
            try:
                process_single_image(model, current_img_path, i)
            except Exception as e:
                print(f"LỖI NGOẠI LỆ tại ảnh {file_name}: {e}")
                continue 
        else:
            # print(f"-> [Cảnh báo] Không tìm thấy file: {file_name}") # Bật lên nếu cần debug
            pass

    print("\n" + "="*40)
    print("🎉 ĐÃ CHẠY XONG TOÀN BỘ DANH SÁCH ẢNH! 🎉")
    print("="*40)