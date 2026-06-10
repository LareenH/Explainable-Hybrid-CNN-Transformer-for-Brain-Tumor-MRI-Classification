Explainable Hybrid CNN--Transformer Framework for Brain Tumor MRI Classification

This repository contains the implementation accompanying the manuscript:

"An Explainable and Externally Validated Hybrid CNN--Transformer Framework for Brain Tumor MRI Classification"

The proposed framework combines convolutional neural networks (CNNs) and Transformer encoder blocks for multiclass brain tumor classification from MRI images. The repository includes all code required to reproduce the experiments presented in the manuscript, including model training, ablation studies, Grad-CAM explainability analysis, and external validation.

Repository Contents
experiment_cnn_transformer_full_pipeline.py
ablation_study_remaining_three_with_chart.py
cross_dataset_generalization.py
gradcam_colab_exact.py
requirements.txt
README.md
Main Scripts
Script	Purpose
experiment_cnn_transformer_full_pipeline.py	Training and evaluation of the proposed hybrid CNN--Transformer model
ablation_study_remaining_three_with_chart.py	Reproduces the ablation study
cross_dataset_generalization.py	External validation on an independent MRI dataset
gradcam_colab_exact.py	Grad-CAM explainability visualizations
Datasets
Primary Dataset

Brain Tumor MRI Dataset

Author: Mohammad Nickparvar

Kaggle:

https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

DOI:

10.34740/KAGGLE/DSV/2645886

The dataset contains approximately 7,000 MRI images distributed across four classes:

Glioma
Meningioma
Pituitary Tumor
No Tumor
External Validation Dataset

Brain Tumor MRI Dataset (Glioma, Meningioma, Pituitary, No Tumor)

Authors:

Md Irfanul Kabir Hira
Md Sohag Hossain
Mst Moriom Akter Bithee

Mendeley Data:

https://data.mendeley.com/datasets/zwr4ntf94j/2

DOI:

10.17632/zwr4ntf94j.2

The dataset contains 12,064 MRI images and was used exclusively for external validation.

Installation

Clone the repository:

git clone https://github.com/YOUR_USERNAME/brain-tumor-cnn-transformer.git

cd brain-tumor-cnn-transformer

Install dependencies:

pip install -r requirements.txt
Dataset Preparation
Kaggle Dataset Structure
BrainTumorMRI/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── pituitary/
│   └── notumor/
│
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── pituitary/
    └── notumor/
External Dataset Structure
External_Mendeley/
├── glioma/
├── meningioma/
├── pituitary/
└── notumor/
Reproducing the Experiments
1. Train the Proposed Model
python experiment_cnn_transformer_full_pipeline.py

Outputs:

Trained model
Accuracy/loss curves
Confusion matrix
Classification report
2. Run the Ablation Study
python ablation_study_remaining_three_with_chart.py

This reproduces:

Transformer-only model
CNN-only model
Hybrid without positional embeddings
Full hybrid model
3. Generate Grad-CAM Visualizations
python gradcam_colab_exact.py

Outputs:

Grad-CAM heatmaps
Overlay visualizations
4. Perform External Validation
python cross_dataset_generalization.py \
    --model_path best_model.keras \
    --class_indices_path class_indices.json \
    --external_data_root path/to/External_Mendeley \
    --output_dir external_results

Outputs:

External confusion matrix
External classification report
Cross-dataset evaluation metrics
Model Configuration
Parameter	Value
Input Size	224 × 224 × 3
CNN Blocks	4
CNN Channels	32 / 64 / 128 / 256
Token Count	196
Embedding Dimension	128
Transformer Blocks	2
Attention Heads	2
FFN Dimension	256
Dropout	0.25
Label Smoothing	0.08
Optimizer	Adam
Learning Rate	1e-4
Reproducibility

All experiments were originally developed and executed using:

Python 3
TensorFlow 2.x
Google Colab
NVIDIA Tesla T4 GPUs

The same preprocessing pipeline was used across all experiments. External validation was performed without retraining, fine-tuning, or domain adaptation.
