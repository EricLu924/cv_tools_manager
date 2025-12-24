import os
import shutil
import random
import kagglehub
import torch
import numpy as np
from PIL import Image

# ==========================================
# 工具函式庫 (Utility Functions)
# ==========================================

def run_download_kaggle_dataset():
    print("--- 執行 Kaggle 資料集下載 ---")
    try:
        # 下載數據集 (請填入實際的 Dataset 名稱)
        path = kagglehub.dataset_download("XXX")

        # 目標路徑
        target_dir = "/home/"
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        # 移動資料到指定目錄
        print(f"正在移動檔案至: {target_dir}")
        for item in os.listdir(path):
            source = os.path.join(path, item)
            destination = os.path.join(target_dir, item)
            
            # 優化：使用 shutil.move 的預設行為處理移動，提升效率
            if os.path.exists(destination):
                print(f"警告: 目標位置已有 {item}，將被覆蓋或發生錯誤")
            
            shutil.move(source, destination)

        print("Path to dataset files:", target_dir)
        
    except Exception as e:
        print(f"下載或移動過程中發生錯誤: {e}")


def run_check_gpu():
    print("--- 執行 GPU 環境檢查 ---")
    
    # 檢查 CUDA 是否可用
    if torch.cuda.is_available():
        print("✅ CUDA 可用")
        print(f"GPU 數量：{torch.cuda.device_count()}")
        # 增加保護：避免在無 GPU 時存取 index 0 導致錯誤
        if torch.cuda.device_count() > 0:
            print(f"GPU 名稱：{torch.cuda.get_device_name(0)}")
    else:
        print("❌ CUDA 不可用")

    # 驗證 Tensor 是否在 GPU 上運算
    tensor = torch.tensor([1.0, 2.0, 3.0])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tensor = tensor.to(device)

    print(f"Tensor 裝置位置：{tensor.device}")


def run_data_augmentation():
    print("--- 執行資料增強 (Data Augmentation) ---")

    def augment_image(image):
        """對圖片應用隨機增強處理。"""
        # 隨機翻轉
        if random.choice([True, False]):
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # 隨機旋轉
        angle = random.choice([0, 90, 180, 270])
        image = image.rotate(angle)

        # 隨機裁剪
        width, height = image.size
        crop_size = random.uniform(0.8, 1.0)
        new_width = int(width * crop_size)
        new_height = int(height * crop_size)

        left = random.randint(0, width - new_width)
        upper = random.randint(0, height - new_height)
        right = left + new_width
        lower = upper + new_height

        image = image.crop((left, upper, right, lower))
        image = image.resize((width, height))

        # 隨機加入雜訊 (優化：確保型別轉換效率)
        image_np = np.array(image)
        noise = np.random.normal(0, 10, image_np.shape).astype(np.int16)
        image_np = np.clip(image_np + noise, 0, 255).astype(np.uint8)
        image = Image.fromarray(image_np)

        return image

    def balance_dataset(base_path, target_count=None):
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        
        if not os.path.exists(base_path):
            print(f"錯誤：路徑 {base_path} 不存在。")
            return

        grade_folders = [os.path.join(base_path, folder) for folder in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, folder))]

        if not grade_folders:
            print("未發現子資料夾。")
            return

        image_counts = {folder: len([img for img in os.listdir(folder) if os.path.splitext(img)[-1].lower() in valid_extensions]) for folder in grade_folders}
        
        # 避免空資料夾導致 max() 錯誤
        if not image_counts:
            print("資料夾內無有效圖片。")
            return
            
        max_count = max(image_counts.values())
        target_count = target_count if target_count is not None else max_count

        print(f"平衡前的圖片數量：{image_counts}")
        print(f"目標平衡到的圖片數量：{target_count}")

        for folder in grade_folders:
            current_count = image_counts[folder]
            if current_count < target_count:
                images = [os.path.join(folder, img) for img in os.listdir(folder) if os.path.splitext(img)[-1].lower() in valid_extensions]
                if not images:
                    print(f"警告：資料夾 {folder} 中沒有有效圖片，跳過增強。")
                    continue

                while current_count < target_count:
                    image_path = random.choice(images)
                    try:
                        with Image.open(image_path) as image:
                            # 複製圖片以避免鎖定原始檔案，並轉換為 RGB 確保相容性
                            image = image.convert("RGB")
                            augmented_image = augment_image(image)
                            new_image_name = f"aug_{current_count}.png"
                            augmented_image.save(os.path.join(folder, new_image_name))
                    except Exception as e:
                        print(f"處理圖片 {image_path} 時發生錯誤: {e}")
                    
                    current_count += 1

        image_counts = {folder: len(os.listdir(folder)) for folder in grade_folders}
        print(f"平衡後的圖片數量：{image_counts}")

    # 使用範例參數 (保留原始命名與設定)
    base_path = r"\home\Dataset" 
    target_count = 1000
    balance_dataset(base_path, target_count)


def run_rename_files():
    print("--- 執行檔案重新命名 (Rename) ---")

    def sort_photo_names(folder_path):
        try:
            if not os.path.exists(folder_path):
                print(f"路徑不存在: {folder_path}")
                return

            file_names = os.listdir(folder_path)
            photo_files = [f for f in file_names if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
            sorted_files = sorted(photo_files)

            for index, file_name in enumerate(sorted_files):
                new_name = f"{index + 1:03d}{os.path.splitext(file_name)[1]}"
                old_path = os.path.join(folder_path, file_name)
                new_path = os.path.join(folder_path, new_name)
                
                # 避免覆蓋自己 (例如已經是 001.jpg)
                if old_path != new_path:
                    os.rename(old_path, new_path)
                    print(f"{file_name} -> {new_name}")

            print("照片重新命名完成！")

        except Exception as e:
            print(f"發生錯誤：{e}")

    # 指定目錄路徑 (保留原始命名)
    folder_path = "Dataset\XXX"
    sort_photo_names(folder_path)


def run_image_merge():
    print("--- 執行圖片合併 (Image Merge) ---")
    
    # 設定變數 (保留原始命名)
    folder_path = r"XXX" 
    image_size = (200, 200)

    try:
        if not os.path.exists(folder_path):
            print(f"路徑不存在: {folder_path}")
            return

        image_paths = []
        for category in os.listdir(folder_path):
            category_path = os.path.join(folder_path, category)
            if os.path.isdir(category_path):
                images = [os.path.join(category_path, file) for file in os.listdir(category_path) if file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif'))]
                image_paths.extend(images)

        # 安全檢查：確保有足夠的圖片
        if len(image_paths) < 40:
            print(f"圖片數量不足 40 張 (目前: {len(image_paths)})，無法執行合併。")
            return

        selected_images = random.sample(image_paths, 40)

        resized_images = []
        for img_path in selected_images:
            # 使用 with 確保檔案關閉
            with Image.open(img_path) as img:
                img = img.convert("RGB") # 確保格式一致
                img_resized = img.resize(image_size)
                resized_images.append(img_resized)

        width, height = image_size
        result = Image.new('RGB', (width * 10, height * 4))

        for i in range(4):
            for j in range(10):
                result.paste(resized_images[i * 10 + j], (j * width, i * height))

        print("正在顯示並儲存結果...")
        # 顯示或儲存結果
        # result.show() # 在伺服器環境可能需要註解掉
        result.save("output_image.jpg")
        print("已儲存為 output_image.jpg")
        
    except Exception as e:
        print(f"合併過程中發生錯誤: {e}")


# ==========================================
# 主程式邏輯 (Switch-Case Implementation)
# ==========================================

def main():
    """
    主控制選單，使用 Python match-case 進行分流。
    """
    print("=========================================")
    print("      Basic Python Tools for CV          ")
    print("=========================================")
    print("1. Download Kaggle Dataset")
    print("2. Check GPU Status")
    print("3. Data Augmentation (Balance Dataset)")
    print("4. Rename Files (Batch)")
    print("5. Image Merge (Grid View)")
    print("0. Exit")
    print("=========================================")
    
    choice = input("請輸入選項 (0-5): ").strip()

    # 使用 Python 3.10+ 的 match-case 結構
    match choice:
        case "1":
            run_download_kaggle_dataset()
        case "2":
            run_check_gpu()
        case "3":
            run_data_augmentation()
        case "4":
            run_rename_files()
        case "5":
            run_image_merge()
        case "0":
            print("程式結束。")
        case _:
            print("無效的輸入，請重新執行並輸入 0-5 之間的數字。")

if __name__ == "__main__":
    main()