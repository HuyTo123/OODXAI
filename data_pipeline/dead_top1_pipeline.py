import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision import models
import numpy as np
import cv2
import shap
from PIL import Image
from skimage.segmentation import slic
from tqdm import tqdm 

# ==============================================================================
# 1. CẤU HÌNH HỆ THỐNG
# ==============================================================================
print("--- KHỞI ĐỘNG HỆ THỐNG THÍ NGHIỆM (FIXED INDEX) ---")

MODEL_PATH = 'testmodel/cat_dogs_huggingface/cat_dog_resnet50_best.pth'
DATA_DIR = "testmodel/cat_dogs_huggingface/Cat_and_Dog_Images/test/Cat"

FILE_BASELINE = "log_baseline.txt"
FILE_MASKED = "log_masked.txt"
FILE_LOST = "log_lost.txt"

START_ID = 0
END_ID = 100

NUM_SUPERPIXELS = 50
BATCH_SIZE = 20
NSAMPLES_MODE = 'auto' # SHAP tự tính toán số mẫu tối ưu

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# 2. UTILS
# ==============================================================================

def init_log_files():
    with open(FILE_BASELINE, 'w') as f: pass
    with open(FILE_MASKED, 'w') as f: pass
    with open(FILE_LOST, 'w') as f: pass
    print("-> Đã khởi tạo lại các file log (.txt)")

def write_log(filename, data_list):
    str_list = ",".join(map(str, data_list[-1])) if len(data_list[-1]) > 0 else "None"
    prefix = " ".join(map(str, data_list[:-1]))
    line = f"{prefix} {str_list}\n"
    with open(filename, 'a') as f:
        f.write(line)

def load_model(device):
    model = models.resnet50(weights=None) 
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2) 
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    else:
        print("Warning: Dùng random weights.")
    model = model.to(device)
    model.eval()
    return model

def transform_masked_image(numpy_img):
    tensor_img = torch.from_numpy(numpy_img.transpose(2, 0, 1)).float()
    return transforms.functional.normalize(tensor_img, mean, std)

# ==============================================================================
# 3. LOGIC DỰ ĐOÁN
# ==============================================================================

def prediction_logic(z, segments_slic, image_numpy_0_1, model, device):
    all_logits = []
    unique_labels = np.unique(segments_slic)
    background_color = image_numpy_0_1.mean((0,1))
    
    for i in range(0, z.shape[0], BATCH_SIZE):
        z_batch = z[i:i + BATCH_SIZE]
        masked_images_np = []
        for mask in z_batch:
            temp_image = image_numpy_0_1.copy()
            inactive_indices = np.where(mask == 0)[0]
            if len(inactive_indices) > 0:
                inactive_labels = unique_labels[inactive_indices]
                mask_all_inactive = np.isin(segments_slic, inactive_labels)
                temp_image[mask_all_inactive] = background_color
            masked_images_np.append(temp_image)

        if len(masked_images_np) > 0:
            tensors = torch.stack([transform_masked_image(img) for img in masked_images_np]).to(device)     
            with torch.no_grad():
                logits = model(tensors)
                all_logits.append(logits.cpu().numpy())
    
    return np.concatenate(all_logits, axis=0)

def prediction_logic_noise(z, segments_slic, image_numpy_0_1, model, device, blocked_label):
    all_logits = []
    unique_labels = np.unique(segments_slic)
    background_color = image_numpy_0_1.mean((0,1))
    
    block_idx = np.where(unique_labels == blocked_label)[0]
    z_noise = z.copy()
    if len(block_idx) > 0:
        z_noise[:, block_idx[0]] = 0 

    for i in range(0, z_noise.shape[0], BATCH_SIZE):
        z_batch = z_noise[i:i + BATCH_SIZE]
        masked_images_np = []
        for mask in z_batch:
            temp_image = image_numpy_0_1.copy()
            inactive_indices = np.where(mask == 0)[0]
            if len(inactive_indices) > 0:
                inactive_labels = unique_labels[inactive_indices]
                mask_all_inactive = np.isin(segments_slic, inactive_labels)
                temp_image[mask_all_inactive] = background_color
            masked_images_np.append(temp_image)

        if len(masked_images_np) > 0:
            tensors = torch.stack([transform_masked_image(img) for img in masked_images_np]).to(device)     
            with torch.no_grad():
                logits = model(tensors)
                all_logits.append(logits.cpu().numpy())

    return np.concatenate(all_logits, axis=0)

# ==============================================================================
# 4. MAIN
# ==============================================================================
if __name__ == '__main__':
    init_log_files()
    model = load_model(DEVICE)
    
    for img_id in tqdm(range(START_ID, END_ID), desc="Tiến độ thí nghiệm", unit="ảnh"):        
        img_name = f"Cat_{img_id}.png"
        full_path = os.path.join(DATA_DIR, img_name)
        
        if not os.path.exists(full_path):
            continue 
            
        print(f"\n>>> Đang xử lý: {img_name}")
        
        img = cv2.imread(full_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        transform_resize = transforms.Compose([
            transforms.Resize((224, 224)),
        ])
        pil_img = Image.fromarray(img)
        pil_img = transform_resize(pil_img)
        image_numpy_0_1 = np.array(pil_img) / 255.0
        
        segments_slic = slic(image_numpy_0_1, n_segments=NUM_SUPERPIXELS, compactness=10, sigma=1, start_label=0)
        unique_segments = np.unique(segments_slic)
        num_segs = len(unique_segments)
        
        inp = transform_masked_image(image_numpy_0_1).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            preds = torch.softmax(model(inp), dim=1)
            predict_score = torch.argmax(preds).item() 
        
        print(f"   Class dự đoán: {predict_score}")

        # --- STEP 1: BASELINE ---
        def step1_runner(z):
            return prediction_logic(z, segments_slic, image_numpy_0_1, model, DEVICE)

        explainer1 = shap.KernelExplainer(step1_runner, np.zeros((1, num_segs)))
        shap_values_1 = explainer1.shap_values(np.ones((1, num_segs)), nsamples=NSAMPLES_MODE, silent=True)
        
        # === SỬA LỖI Ở ĐÂY ===
        # Shape: [1, n_segments, n_classes] -> Lấy [0, :, predict_score]
        vals_1 = shap_values_1[0, :, predict_score]
        
        positive_indices = np.where(vals_1 > 0)[0]
        if len(positive_indices) == 0:
            print("   [Skip] Không có segment tích cực.")
            continue

        val_idx_pairs = []
        for idx in positive_indices:
            val_idx_pairs.append((vals_1[idx], unique_segments[idx]))
        
        val_idx_pairs.sort(key=lambda x: x[0], reverse=True)
        sorted_segment_labels_1 = [idx for val, idx in val_idx_pairs]
        
        best_segment = sorted_segment_labels_1[0]
        
        write_log(FILE_BASELINE, [img_id, predict_score, sorted_segment_labels_1])
        print(f"   [Step 1] Best: {best_segment}")

        # --- STEP 2: MASKED ---
        def step2_runner(z):
            return prediction_logic_noise(z, segments_slic, image_numpy_0_1, model, DEVICE, blocked_label=best_segment)
        
        explainer2 = shap.KernelExplainer(step2_runner, np.zeros((1, num_segs)))
        shap_values_2 = explainer2.shap_values(np.ones((1, num_segs)), nsamples=NSAMPLES_MODE, silent=True)
        
        # === SỬA LỖI Ở ĐÂY ===
        vals_2 = shap_values_2[0, :, predict_score]
        
        positive_indices_2 = np.where(vals_2 > 0)[0]
        val_idx_pairs_2 = []
        for idx in positive_indices_2:
            val_idx_pairs_2.append((vals_2[idx], unique_segments[idx]))
        
        val_idx_pairs_2.sort(key=lambda x: x[0], reverse=True)
        sorted_segment_labels_2 = [idx for val, idx in val_idx_pairs_2]
        
        write_log(FILE_MASKED, [img_id, predict_score, sorted_segment_labels_2])

        # --- STEP 3: LOST ---
        K_TOP = 4
        lost_segments = []
        original_top_k = sorted_segment_labels_1[:K_TOP]
        for seg in original_top_k:
            if seg not in sorted_segment_labels_2:
                lost_segments.append(seg)
        
        write_log(FILE_LOST, [img_id, best_segment, lost_segments])
        print(f"   [Done] Lost segments: {len(lost_segments)}")

    print("\n--- HOÀN TẤT ---")