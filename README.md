# 🎨 Art Style Classification

This repository contains code and models for classifying paintings into their respective genres using deep learning. The project utilizes the **WikiArt dataset**, which consists of approximately **80,000 paintings** across **27 different art genres**.

## 📌 Project Overview
The goal of this project is to develop machine learning models that can **automatically classify paintings** based on their artistic style. The models explore **various architectures**, ranging from simple **softmax classifiers** to **deep convolutional neural networks (CNNs)** and **pre-trained models like ResNet and EfficientNet**.

## 📂 Repository Structure
```
📦 Art_Style_Classification
├── Baseline_Model.ipynb          # Simple baseline model predicting the most common class
├── SoftMax.ipynb                 # Basic softmax model for multi-class classification
├── Fully_Connected_NN_1.ipynb     # Simple fully connected neural network
├── Fully_Connected_NN_2.ipynb     # Deeper fully connected neural network
├── fully_connected_NN_3.ipynb     # Extended fully connected network with batch normalization
├── CNN1.ipynb                     # Convolutional Neural Network (CNN) Model 1
├── CNN2.ipynb                     # Convolutional Neural Network (CNN) Model 2
├── cnn3.ipynb                      # Convolutional Neural Network (CNN) Model 3
├── CNN_For_5_Classes.ipynb         # CNN model trained for a simplified 5-class classification
├── Resnet.ipynb                    # ResNet-based feature extraction model
├── resnet50.ipynb                   # Fine-tuned ResNet50 model
├── Eff_fully_connected_nn.ipynb     # EfficientNet feature extraction model
├── HyperNetwork.ipynb               # Hypernetwork approach for dynamic weight learning
├── Preprocessing.ipynb               # Data preprocessing techniques for class balancing
├── Project Notebook.pdf              # Detailed documentation of the project
└── README.md                         # This file
```

## 📊 Dataset
- **Source:** WikiArt dataset (~80,000 paintings)
- **Genres Classified:** 13 (after preprocessing)
- **Features Considered:**
  - Colors and shades
  - Painting tools and techniques
  - Overall composition
  - Artist and objects depicted
- **Data Preprocessing:**
  - Classes with fewer than **2,000 images were removed**.
  - **Data augmentation** (flipping, rotation, brightness adjustments) applied to underrepresented classes.
  - **Classes with >5,000 images were downsampled** to balance the dataset.
  - Images resized to **256×256 pixels** for consistency.

## 🏗️ Model Architectures
1. **Baseline Model:** Predicts the majority class, achieving ~7% accuracy.
2. **SoftMax Regression Model:** A simple linear model to classify paintings.
3. **Fully Connected Neural Network (FNN):** Multi-layer perceptron with different depths.
4. **Convolutional Neural Networks (CNNs):** Standard CNN architectures with increasing complexity.
5. **Pre-trained Models:**
   - **ResNet18 & ResNet50:** Feature extraction using pre-trained ResNet models.
   - **EfficientNet-B0:** Feature extraction for efficient classification.
6. **Hypernetwork Model:** A novel architecture using a learnable latent vector to generate weights dynamically.

## 📈 Results & Performance
| Model                           | Accuracy |
|---------------------------------|----------|
| Baseline Model                  | ~7%      |
| Softmax Model                    | ~12%     |
| Fully Connected NN               | ~25%     |
| CNN (Best Model)                 | ~40%     |
| ResNet18                         | ~55%     |
| ResNet50                         | **73%**  |
| EfficientNet-B0                   | 60.6%    |
| Hypernetwork (ResNet50-based)     | 71.7%    |

## 🔍 Key Findings
- **CNNs significantly outperformed fully connected networks** due to their ability to capture spatial features.
- **Pre-trained models like ResNet** improved classification accuracy, demonstrating the power of transfer learning.
- **Hypernetwork models** introduced an interesting approach but did not surpass the fine-tuned ResNet50 model.
- **Five-class classification performed better** than full multi-class classification due to reduced complexity.

## 🚀 How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/AviShatkberg/Art_Style_Classification.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
4. Run `Preprocessing.ipynb` to prepare the dataset.
5. Train models by running `Baseline_Model.ipynb`, `CNN1.ipynb`, `Resnet.ipynb`, etc.

## 🛠️ Future Improvements
- Experiment with **Vision Transformers (ViTs)** for improved classification.
- Implement **GAN-based augmentation** for better generalization.
- Fine-tune hypernetwork architecture for improved performance.

## 📜 Citation
If you use this code, please cite:
```
@misc{artstyleclassification2025,
  author = {Avi Shatkberg},
  title = {Art Style Classification using Deep Learning},
  year = {2025},
  url = {https://github.com/AviShatkberg/Art_Style_Classification}
}
```

## 🤝 Acknowledgements
- WikiArt for the dataset
- PyTorch & TensorFlow for model implementations
- ResNet & EfficientNet architectures for pre-trained models
