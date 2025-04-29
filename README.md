# Thermal Pistol Detection using YOLOv11

This project focuses on detecting thermal pistols in images using a fine-tuned YOLOv11 model. The model is trained on a custom dataset containing thermal images of pistols, split into training, validation, and test sets. The repository includes code for training, evaluation, and inference, along with visualizations of model performance.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model](#model)
- [Setup](#setup)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [Results](#results)
- [License](#license)

## Overview
The goal of this project is to detect thermal pistols in images using a YOLOv11 model. The model is fine-tuned on a custom dataset and evaluated for accuracy, precision, recall, and F1 score. The repository includes scripts for training, validation, and inference, as well as visualizations of the dataset and model performance.

## Dataset
The dataset consists of thermal images of pistols, split into three subsets:
- **Train**: 80% of the data
- **Validation**: 10% of the data
- **Test**: 10% of the data

Each image is accompanied by a corresponding label file in YOLO format, containing bounding box annotations.

### Dataset Structure

## Model
The model used in this project is **YOLOv11n**, a lightweight version of the YOLOv11 architecture. The model is fine-tuned on the custom thermal pistol dataset.

### Key Metrics
- **mAP50-95**: 0.7281
- **Precision**: 1.0000
- **Recall**: 0.9964
- **F1 Score**: 0.9982

## Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/Alishan45/thermal-pistol-detection.git
   cd thermal-pistol-detection
pip install -r requirements.txt
python train.py --data /path/to/dataset.yaml --epochs 100 --imgsz 640
python val.py --data /path/to/dataset.yaml --weights /path/to/best.pt
python detect.py --weights /path/to/best.pt --source /path/to/image.jpg

---

### **Key Points in the README**
1. **Overview**:
   - Briefly explains the purpose of the project and the tools used.

2. **Dataset**:
   - Describes the dataset structure and format.

3. **Model**:
   - Provides details about the YOLOv11n model and its performance metrics.

4. **Setup**:
   - Includes clear instructions for setting up the project.

5. **Training, Evaluation, and Inference**:
   - Provides commands for training, validating, and running inference.

6. **Results**:
   - Shows visualizations of training metrics, confusion matrix, PR curve, and sample predictions.

7. **License**:
   - Specifies the MIT License for open-source use.

---

### **Additional Notes**
- Replace `/path/to/` placeholders with actual paths in your repository.
- Ensure the images linked in the **Results** section (e.g., `results.png`, `confusion_matrix.png`) are generated and saved in the correct directory.
- Update the GitHub repository link (`https://github.com/your-username/thermal-pistol-detection.git`) with your actual repository URL.

This README file is professional, easy to follow, and provides all the necessary information for users to understand and use your project. Let me know if you need further assistance!
