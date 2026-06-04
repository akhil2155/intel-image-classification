# Intel Image Classification using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-2.0%2B-red.svg)](https://keras.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Computer Vision and Deep Learning pipeline utilizing Convolutional Neural Networks (CNNs) to automate multi-class scene categorization across diverse geographic environments. The project explores data preprocessing, structural dataset pipeline optimization, structural model architecture configuration, and comparative evaluation metrics.

---

## 📌 Project Overview
Automated scene classification is foundational for geospatial analysis, autonomous vehicle navigation, and remote sensing. This project builds a localized image recognition engine capable of processing raw structural imagery and classifying it into six distinct environmental landscapes with high precision.

### Target Classes:
* 🏢 **Buildings** (Urban infrastructure)
* 🌲 **Forest** (Dense vegetation and woodlands)
* 🏔️ **Glacier** (Ice caps and frozen terrains)
* ⛰️ **Mountain** (Rock formations and high altitudes)
* 🌊 **Sea** (Open water bodies and coastal bounds)
* 🛣️ **Street** (Roadways and urban pathways)

---

## 🛠️ Tech Stack & Core Libraries
* **Core Language:** Python
* **Deep Learning Frameworks:** TensorFlow, Keras
* **Scientific Computing & Data Prep:** NumPy, Scikit-learn, Scikit-image
* **Data Visualization & Analytics:** Matplotlib, Seaborn, Evaluation Confusion Matrices

---

## 🧬 Machine Learning Pipeline Architecture

The pipeline processes raw pixels through structured functional blocks to establish an end-to-end computer vision workflow:

1. **Data Ingestion & Formatting:** Images are parsed from distinct local directories (`seg_train`, `seg_test`, `seg_pred`).
2. **Preprocessing & Tensor Transformation:** High-resolution imagery is downsampled to a standardized target shape (e.g., $150 \times 150$ pixels) and min-max pixel values are scaled down to a $[0, 1]$ normal range to accelerate gradient descent stability.
3. **Dataset Pipeline Optimization:** Data streams are shuffled dynamically to mitigate selection bias and batched safely to prevent local hardware out-of-memory errors.
4. **CNN Model Instantiation:** Stacking structural multi-layer 2D Convolutions (`Conv2D`), local pool maximizations (`MaxPooling2D`), Dropout regularization layers to mitigate overfitting, and multi-neuron Dense classification heads.
5. **Evaluation Strategy:** Validated against a separated, unexposed validation slice (`seg_test`) utilizing Categorical Cross-Entropy loss maps and structural Precision-Recall confusion matrices.

---

## 📊 Model Architecture Design
While configurations can be modified dynamically inside the Jupyter Notebook, the network maps the following standard architecture hierarchy:
[Input Layer: 150x150x3 RGB Images]
│
[Conv2D + ReLU] ──────> Extraction of low-level edges & textures
│
[MaxPooling2D (2x2)] ───> Dimensional reduction & translation invariance
│
[Conv2D + MaxPooling2D] ───> Intermediate structural extraction
│
[Flatten Layer] ──────> Linear conversion to dense feature vector
│
[Dense Hidden Head] ────> High-level abstraction learning
│
[Dropout (0.5)] ───────> Regularization to force model robust generalization
│
[Dense Output (6 Units)] ───> Softmax activation projecting class likelihoods

## 🚀 How To Run This Project Locally

### 1. Prerequisites & Environment Setup
Clone this repository to your local computer and verify your directory configuration:
bash
git clone [https://github.com/akhil2155/intel-image-classification.git](https://github.com/akhil2155/intel-image-classification.git)
cd "intel-image-classification"

Ensure your directory contains the following structural layout matching the configuration expected by the notebook:
├── Intel Image Classification.ipynb
├── .gitignore
├── seg_train/
│   └── seg_train/ (buildings, forest, glacier, mountain, sea, street)
├── seg_test/
│   └── seg_test/  (buildings, forest, glacier, mountain, sea, street)
└── seg_pred/
    └── seg_pred/  (unlabeled prediction images)

2. Install Dependency Ecosystem
Run the following command to make sure you have the required packages installed:

pip install tensorflow keras numpy matplotlib seaborn scikit-learn

3. Initialize the Pipeline
Fire up Jupyter Lab or Jupyter Notebook to interact with the environment:

jupyter notebook

Open Intel Image Classification.ipynb and execute the cell layers sequentially to ingest the data, build the layers, run training backpropagation, and evaluate your final performance metrics.

📈 Future Scope & Experimental Roadmaps
Transfer Learning Integration: Swap out the custom baseline CNN framework with proven deep learning feature extractors like ResNet50, MobileNetV2, or EfficientNet pretrained weights to radically minimize local training epoch times while yielding exceptional accuracy gains.

Data Augmentation Deployments: Implement random vertical/horizontal flips, rotations, and zoom operations mid-stream to artificially multiply training diversity and combat validation loss divergence.

Hyperparameter Tuning: Automate learning rate schedule drops, try alternative optimizers (e.g., Adamax, RMSprop), and configure custom bottleneck filters using Keras Tuner tracking.

📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
