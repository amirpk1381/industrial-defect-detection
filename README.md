# Automated Industrial Defect Detection using Deep Learning

A high-precision Deep Learning pipeline designed for automated quality control and structural defect inspection. This project implements a custom Convolutional Neural Network (CNN) architecture to classify anomalies and utilizes Explainable AI (XAI) techniques to interpret the model's decision-making process.

## 📊 Key Highlights & Performance
 High Accuracy: Achieved 99.1% Accuracy on the test dataset.
 *Explainable AI (XAI): Integrated Grad-CAM (Gradient-weighted Class Activation Mapping) using PyTorch forward/backward hooks to visualize and audit the model's focus area.
 *Robust Evaluation: Evaluated extensively using a Confusion Matrix to track and minimize false passes (critical for industrial standards).

---

## 🏗️ Architecture & Technical Workflow

1. Feature Extraction: Built using advanced CNN backbones (`EfficientNet` / `ResNet`) implemented from scratch/fine-tuned in PyTorch.
2. Feature Map Interception: Registered custom PyTorch hooks on the deep convolutional blocks (`block5` / `block6`) to capture spatial feature maps and gradients during the backward pass.
3. Heatmap Generation: Computed the weighted combination of forward activation maps using backpropagated gradients to generate raw localization maps.
4. Post-Processing: Applied `cv2.resize` (with bilinear interpolation) and `np.clip` to smoothly overlay the activation maps directly onto the input images.

---

## 📈 Results and Visualizations

### 1. Confusion Matrix Analysis
The model demonstrates near-perfect classification, isolating defective products from nominal ones with minimal error.

 *True Negatives (Defective classified as Defective): 363 items
 *True Positives (OK classified as OK): 227 items
 *False Passes: Only 5 items out of 595 total test samples.

### 2. Grad-CAM Interpretation
By visualizing the last convolutional layers, the heatmaps confirm that the network successfully locks onto the actual physical geometric anomalies and surface cracks of the components, rather than relying on background noise or lighting conditions.

---

## 🛠️ Requirements & Environment
 Python 3.x
 PyTorch (torch, torchvision)
 OpenCV (opencv-python)
 Scikit-Learn
 Seaborn & Matplotlib
* NumPy

