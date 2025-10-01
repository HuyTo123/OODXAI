import cv2
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage.segmentation import mark_boundaries
import numpy as np
from scipy.ndimage import center_of_mass
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path

# --- 1. CÁC THAM SỐ TÙY CHỈNH ---

# Đường dẫn đến ảnh của bạn
IMAGE_PATH = "testmodel/cat_dogs_huggingface/Cat_and_Dog_Images/test/Cat/Cat_9.png"
input_path = Path(IMAGE_PATH)
filename = input_path.name  # Lấy ra 'Cat_0.png'
class_name = input_path.parts[-2] # Lấy ra 'Cat'

output_base_dir = Path("testmodel/cat_dogs_huggingface/CatandDog_segment_noise")
output_class_dir = output_base_dir / class_name
output_class_dir.mkdir(parents=True, exist_ok=True)
output_path = output_class_dir / filename

# Các tham số cho thuật toán SLIC
NUM_SUPERPIXELS = 50
COMPACTNESS = 10
SIGMA = 1

# ===> Nơi bạn tùy chỉnh các vùng cần bôi đen <===
LABELS_TO_BLACK = [ 20, 11, 26]

# --- 2. XỬ LÝ VÀ PHÂN VÙNG ẢNH ---

# Tải ảnh bằng OpenCV
image = cv2.imread(IMAGE_PATH)

if image is None:
    print(f"Lỗi: Không thể tìm thấy hoặc đọc được ảnh từ đường dẫn: {IMAGE_PATH}")
else:
    # Chuyển đổi hệ màu từ BGR (OpenCV) sang RGB (matplotlib, scikit-image)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Định nghĩa chuỗi xử lý ảnh giống hệt OodKernalSHAP.p
    transform_pipeline = transforms.Compose([
        transforms.Resize((224, 224)), # Resize ảnh về 224x224
        transforms.ToTensor()          # Chuyển thành Tensor và scale về [0, 1]
    ])

    # Áp dụng chuỗi xử lý
    pil_image = Image.fromarray(image_rgb)
    image_tensor = transform_pipeline(pil_image)
    image_processed = image_tensor.permute(1, 2, 0).numpy()

    # Áp dụng SLIC trên ảnh đã qua xử lý (lần đầu)
    segments_slic = slic(image_processed, n_segments=NUM_SUPERPIXELS,
                         compactness=COMPACTNESS, sigma=SIGMA, start_label=0)

    # --- 3. THAY ĐỔI MÀU SẮC CÁC VÙNG ĐÃ CHỌN ---

    # Tạo một bản sao của ảnh gốc để chỉnh sửa
    output_image = image_processed.copy()

    # Lặp qua danh sách các label bạn muốn đổi thành màu đen
    for label in LABELS_TO_BLACK:
        mask = segments_slic == label
        output_image[mask] = [0, 0, 0] # Màu đen
    image_to_save_uint8 = (output_image * 255).astype(np.uint8)
    pil_image_to_save = Image.fromarray(image_to_save_uint8)
    pil_image_to_save.save(output_path)
    print(f"Ảnh đã được lưu tại: {output_path}")
    # --- 4. HIỂN THỊ HÌNH ẢNH (ĐÃ TỐI ƯU) ---

    # Tạo một cửa sổ hiển thị lớn hơn với 2 ảnh con
    fig, ax = plt.subplots(1, 2, figsize=(15, 8), sharex=True, sharey=True)

    # --- Xử lý Ảnh 1: Ranh giới và Label trên ảnh gốc ---
    ax[0].imshow(mark_boundaries(image_processed, segments_slic))
    ax[0].set_title("Ảnh Gốc & Superpixel")
    ax[0].axis('off')

    unique_labels = np.unique(segments_slic)
    
    for label in unique_labels:
        y, x = center_of_mass(segments_slic, segments_slic, label)
        ax[0].text(x, y, str(label), 
                   fontsize=14, 
                   color='Red', 
                   ha='center',
                   va='center')

    # --- Xử lý Ảnh 2: Chạy lại SLIC và hiển thị kết quả ---
    
    # *** BẮT ĐẦU THAY ĐỔI ***
    # 1. Chạy lại thuật toán SLIC trên ảnh đã bị bôi đen
    segments_slic_after_blackout = slic(output_image, n_segments=NUM_SUPERPIXELS,
                                         compactness=COMPACTNESS, sigma=SIGMA, start_label=0)

    # 2. Hiển thị ranh giới của các superpixel MỚI
    ax[1].imshow(mark_boundaries(output_image, segments_slic_after_blackout))
    ax[1].set_title("Ảnh Bôi Đen & Superpixel Mới")
    ax[1].axis('off')

    # 3. Lấy và hiển thị các label MỚI
    unique_labels_after = np.unique(segments_slic_after_blackout)
    
    for label in unique_labels_after:
        # Dùng center_of_mass trên segmentation MỚI
        y, x = center_of_mass(segments_slic_after_blackout, segments_slic_after_blackout, label)
        
        # Dùng ax[1].text để vẽ số lên ảnh thứ hai
        ax[1].text(x, y, str(label), 
                   fontsize=14, 
                   color='yellow', # Đổi màu để dễ phân biệt
                   ha='center',
                   va='center')
    # *** KẾT THÚC THAY ĐỔI ***

    plt.tight_layout()
    plt.show()