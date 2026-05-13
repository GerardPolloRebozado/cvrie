# MRI Tumor Detection and Analysis

This project leverages Machine Learning and Computer Vision techniques to detect tumors in MRI scans and analyze patient testimonials through unsupervised clustering.

## Features

- **Tumor Detection (Supervised Learning):** Classifies MRI scans into categories such as Healthy, Glioma, Meningioma, Pituitary, or general Tumor using an SVM model.
- **Testimonial Analysis (Unsupervised Learning):** Clusters patient testimonials using NLP (TF-IDF + K-Means) to identify common themes and concerns.
- **Web Interface:** A FastAPI-based web application for easy interaction with the models.
- **Advanced Preprocessing:** Uses adaptive histogram equalization and TV Chambolle denoising to enhance MRI image quality.
- **Feature Extraction:** Implements GLCM (Gray-Level Co-occurrence Matrix) for texture analysis and HOG (Histogram of Oriented Gradients).

## Project Structure

- `scripts/`: Contains the core application logic and training scripts.
  - `api.py`: FastAPI web server.
  - `supervised_training.py`: Script to train the SVM model for tumor detection.
  - `unsupervised_learning.py` / `train_model_unsupervised_learning.py`: Scripts for testimonial clustering.
- `dataset_exploration.ipynb`: Notebook for initial data analysis.
- `supervised_learning.ipynb`: Detailed workflow for tumor detection training.
- `unsupervised_learning.ipynb`: Workflow for testimonial clustering.
- `test_*.ipynb`: Experimental notebooks for various CV techniques (blob detection, corner detection, etc.).
- `templates/`: HTML templates for the web interface.

## Dataset

The MRI dataset consists of 5,000 images:
- **3,000** images showing tumors.
- **2,000** healthy images.

The data is categorized into:
- Healthy
- Glioma
- Meningioma
- Pituitary
- Tumor

##️ Setup & Installation

### Prerequisites

- Python 3.8+
- Recommended: A virtual environment (venv or conda)

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Training the Models

- To train the supervised tumor detection model:
  ```bash
  python scripts/supervised_training.py
  ```
- To train the unsupervised testimonial clustering model:
  ```bash
  python scripts/train_model_unsupervised_learning.py
  ```

### Running the Web Application

Start the FastAPI server:
```bash
python scripts/api.py
```
Or using uvicorn:
```bash
uvicorn scripts.api:app --reload
```
The app will be available at `http://127.0.0.1:8000`.

## Methodology

### MRI Analysis
1. **Preprocessing:** Grayscale conversion, resizing to 256x256, and denoising.
2. **Feature Engineering:** Texture features extracted via GLCM and shape features via HOG.
3. **Classification:** A Support Vector Machine (SVM) classifier trained on the extracted features.

### Testimonial Clustering
1. **Text Cleaning:** Normalization and removal of non-alphabetic characters.
2. **Vectorization:** TF-IDF (Term Frequency-Inverse Document Frequency).
3. **Clustering:** K-Means clustering to group similar testimonials.
