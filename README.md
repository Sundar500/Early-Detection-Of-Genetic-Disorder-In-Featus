# Early Detection of Fetal Genetic Disorders (Multimodal Deep Learning)

## 🌟 Project Overview

This project implements a **Multimodal Deep Learning** system for the **early prediction of fetal genetic disorders** by integrating two heterogeneous data sources: **grayscale ultrasound images** and **structured clinical tabular data**.

The goal is to overcome the limitations of single-modality diagnosis by leveraging a hybrid neural network architecture to enhance prediction accuracy and provide reliable, objective support for prenatal care.

---

## 💡 Key Features and Methodology

* **Dual-Input Architecture:** The core model consists of two parallel streams:
    * **Image Input:** Processed by a **Convolutional Neural Network (CNN)** to extract spatial features (e.g., nuchal translucency, cranial structure) from fetal ultrasound images.
    * **Tabular Data Input:** Processed by **Dense Layers** to interpret numerical features (e.g., maternal age, genetic test scores, symptom severity).
* **Multimodal Fusion:** The feature vectors from the CNN and Dense Layers are concatenated and passed through subsequent Fully Connected Layers for final classification.
* [cite_start]**Classification:** The model classifies the fetus into one of six categories: **Healthy**, **Down Syndrome**, **Turner Syndrome**, **Klinefelter Syndrome**, **Edwards Syndrome**, and **Patau Syndrome**[cite: 70].
* [cite_start]**Explainable AI (XAI):** The system supports the generation of **Grad-CAM heatmaps** to visually indicate which regions of the ultrasound image were most influential in the prediction, enhancing clinician trust and transparency[cite: 693, 673].
* [cite_start]**Real-Time Prediction:** Designed to provide fast, patient-specific predictions through a Gradio web interface[cite: 72, 354].

---

## 🛠️ Technology Stack (System Requirements)

The system relies on a Python-based machine learning environment, utilizing major open-source deep learning frameworks.

### 💻 Software
| Component | Requirement | Role |
| :--- | :--- | :--- |
| **Programming Language** | Python 3.7+ | [cite_start]Core language for development and ML libraries[cite: 420]. |
| **Deep Learning Framework** | TensorFlow 2.x or PyTorch 1.7+ | [cite_start]Infrastructure for building and training the hybrid neural network[cite: 425]. |
| **Data Processing** | NumPy, Pandas, Scikit-learn | [cite_start]Handling tabular data, scaling, encoding, and dataset splitting[cite: 431, 478, 480]. |
| **Image Processing** | OpenCV (`cv2`), Pillow (PIL) | [cite_start]Grayscale conversion, resizing to $128\times128$ pixels, and normalization[cite: 484, 485, 486, 745]. |
| **User Interface (UI)** | Gradio | [cite_start]Deployment interface for real-time inference[cite: 72, 741]. |
| **Development Environment** | Jupyter Notebook/VS Code/Google Colab | [cite_start]Execution and rapid prototyping[cite: 439, 456]. |

### ⚙️ Hardware (Recommended)
* [cite_start]**Processor (CPU):** Intel Core i7/i9 or AMD Ryzen 7/9[cite: 393].
* [cite_start]**RAM (Memory):** 16 GB or higher[cite: 397].
* [cite_start]**GPU:** NVIDIA GPU with CUDA support and at least 4-8 GB VRAM (e.g., GTX 1660, RTX 2060 or better) for faster training[cite: 408].

---

## 📂 Project Structure (Modules)

[cite_start]The system is organized into the following logical modules[cite: 465]:

1.  [cite_start]**Data Preparation (`7.2.1`):** Cleans, scales, and encodes tabular data (using `StandardScaler`, `LabelEncoder`) and converts/resizes images, followed by **Tabular-Image Pairing**[cite: 471, 497].
2.  [cite_start]**Feature Extraction (`7.2.2`):** Extracts features from images using a **CNN** and from tabular data using **Dense Layers**[cite: 499, 503, 512].
3.  [cite_start]**Model Training (`7.2.3`):** Trains the fused dual-input model and evaluates performance metrics (Accuracy, Precision, Recall)[cite: 515, 521].
4.  [cite_start]**Disorder Prediction (`7.2.4`):** Takes new preprocessed inputs (image + tabular features) and performs the final classification using the trained model[cite: 529].
5.  [cite_start]**Report Generation (`7.2.5`):** Creates a structured, human-readable diagnostic report, including the predicted disorder, confidence scores, and Grad-CAM visualizations[cite: 543].

---

## 🚀 Getting Started

### Prerequisites
1.  Python 3.7+ installed.
2.  A development environment (e.g., Visual Studio Code, PyCharm, or Google Colab).

### Installation (using a virtual environment)

```bash
# Clone the repository
git clone <Your-Repo-Link>
cd early-detection-genetic-disorders

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use: .\venv\Scripts\activate

# Install required packages
pip install tensorflow keras pandas numpy opencv-python scikit-learn gradio
