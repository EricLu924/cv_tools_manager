# Computer Vision Data Process Tools Manager
【 Basic Python Tools for CV (Manager Edition) 】

This repository contains a unified **Python utility suite** designed for Computer Vision (CV) and Machine Learning (ML) tasks. 
All tools are integrated into a single script with a menu-driven interface using `switch-case` (match-case) logic.

It is designed to help researchers and engineers quickly execute common tasks like dataset downloading, GPU checking, augmentation, and file management without switching between multiple scripts.

> ⚠️ **Requirement**: Python 3.10 or higher is required to run the `match-case` syntax.

---

## 🚀 How to Use

1.  **Setup Environment**: Ensure you have the required libraries installed:
    ```bash
    pip install torch numpy pillow kagglehub
    ```
2.  **Run the Manager**:
    Execute the main script in your terminal or JupyterLab:
    ```bash
    python cv_tools_manager.py
    ```
3.  **Select a Tool**:
    Enter the number corresponding to the tool you want to run (e.g., enter `2` to check GPU).

---

## 🛠 Tools Included

The following functionalities are integrated into `cv_tools_manager.py`:

### **1. Download Kaggle Dataset**
- **Function**: Downloads datasets directly from KaggleHub and moves them to a target directory.
- **Key Action**: Auto-creates destination folders and handles file moving.

### **2. Check GPU**
- **Function**: Diagnoses system GPU status.
- **Key Action**: Reports CUDA availability, device count, GPU name, and verifies Tensor device allocation.

### **3. Data Augmentation**
- **Function**: Balances datasets by augmenting images in classes with fewer samples.
- **Key Action**: Performs random flip, rotation, crop, and noise addition to reach a `target_count`.

### **4. Rename Files**
- **Function**: Batch renames image files in a directory to a sequential numerical format (e.g., `001.jpg`).
- **Key Action**: Sorts files first to ensure deterministic ordering before renaming.

### **5. Image Merge**
- **Function**: Randomly samples 40 images from a dataset and creates a 4x10 visualization grid.
- **Key Action**: Resizes images to (200, 200) and saves the combined result as `output_image.jpg`.

---

## ⚙️ Configuration
Currently, path variables (e.g., `Dataset\XXX`, `/home/`) are set to default placeholders inside the script functions. 
To use this on your own data, please open `cv_tools_manager.py` and update the variable strings within the respective functions.
