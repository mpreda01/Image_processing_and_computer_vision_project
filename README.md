# Image Processing and Computer Vision Projects

This repository contains two assignment modules for the **Image Processing and Computer Vision** course at the University of Bologna.

---

## 📁 Repository Structure

```
.
├── assignment_module_one_sift.ipynb    # Module 1: Book Recognition with SIFT
├── assignment_module_two_efficientNet.ipynb  # Module 2: Pet Classification with EfficientNet
├── compute_metrics.py                  # Evaluation metrics (IoU, segmentation)
├── extract_ground_truth.py            # Ground truth extraction utilities
├── labels.json                         # Annotated ground truth data
├── results.json                        # Module 1 results
├── grid_search_results.json           # Module 2 hyperparameter search results
├── dataset/
│   ├── models/                        # Reference images for Module 1
│   └── scenes/                        # Shelf images for Module 1 testing
└── README.md
```

---

## 📚 Module 1: Product Recognition of Books (SIFT-based)

**File:** `assignment_module_one_sift.ipynb`

### Objective
Develop a traditional computer vision system that identifies and localizes books on shelves using **Scale-Invariant Feature Transform (SIFT)** and traditional image processing techniques.

### Task Description
Given reference images for each book, the system must identify and localize them within shelf images by computing:

1. **Instance Count**: Number of instances of each book detected
2. **Dimensions**: Bounding box area in pixels for each detected instance
3. **Positions**: Four corners (top-left, top-right, bottom-left, bottom-right) of each bounding box
4. **Visual Output**: Overlaid bounding boxes on the original scene images

### Implementation Approach
- Uses only **traditional computer vision techniques** 
- Leverages SIFT feature detection and matching
- Processes reference models and test scenes located in `dataset/models/` and `dataset/scenes/`

### Evaluation
- Results stored in `results.json`
- Ground truth annotations in `labels.json`
- Metrics computed using `extract_ground_truth.py` and `compute_metrics.py`

---

## 🐕🐈 Module 2: Pet Classification (EfficientNet-based)

**File:** `assignment_module_two_efficientNet.ipynb`

### Objective
Implement and fine-tune a deep learning model to classify pet images into 37 different breeds of cats and dogs.

### Dataset
- **Source**: [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)
- **Dataset Size**: 37 pet breeds 
- **Train/Val/Test Split**: Custom split different from the original dataset

### Implementation Overview

#### Part 1: Custom Neural Network
- Implement a neural network from scratch for image classification
- Develop and train the model without using pretrained weights
- Baseline architecture to understand deep learning fundamentals

#### Part 2: Transfer Learning with EfficientNet
- Fine-tune a pretrained EfficientNet model from PyTorch
- Leverage transfer learning to achieve better performance
- Experimental variations using different configurations

### Key Features
- **Ablation Study**: Tests various hyperparameters and architectural choices
  - Batch normalization effects
  - Learning rate variations 
  - Pooling strategies 
  - Data augmentation impact
  
- **Visualization**: 
  - Confusion matrices
  - Classification reports
  - Sample predictions with confidence scores
  - Training/validation curves

### Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score 
- Confusion matrices for detailed error analysis

### Tools & Frameworks
- **Deep Learning**: PyTorch
- **Experiment Tracking**: Weights & Biases (W&B)
- **Data Processing**: torchvision, Pillow, NumPy
- **Evaluation**: scikit-learn

---

## 🛠️ Utility Functions

### `compute_metrics.py`
Provides evaluation utilities for instance segmentation:
- **`iou_mask()`**: Computes Intersection over Union between binary masks
- **`polygon_to_mask()`**: Converts polygon annotations to binary masks
- **`evaluate_instance_segmentation()`**: Evaluates segmentation quality per class

### `extract_ground_truth.py`
Extracts and processes ground truth annotations:
- **`calculate_polygon_area()`**: Computes polygon area in pixels
- **`extract_ground_truth_areas()`**: Aggregates ground truth data from `labels.json`

---

## 👥 Authors

- **Francesca Conti** - francesca.conti22@studio.unibo.it 
- **Matteo Preda** - matteo.preda2@studio.unibo.it 
