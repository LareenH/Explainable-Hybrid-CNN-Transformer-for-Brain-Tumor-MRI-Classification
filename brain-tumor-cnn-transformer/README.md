# Explainable Hybrid CNN–Transformer Framework for Brain Tumor MRI Classification

This repository contains the implementation of a lightweight hybrid CNN–Transformer framework for multiclass brain tumor classification from MRI images.

The project combines convolutional feature extraction with Transformer-based contextual modeling and includes:

* Training and evaluation of the proposed CNN–Transformer model
* Ablation studies
* Grad-CAM explainability analysis
* Cross-dataset external validation

All experiments were developed and executed using TensorFlow 2.x in Google Colab.

---

## Repository Structure

```text
.
├── experiment_cnn_transformer_full_pipeline.py
├── ablation_study_remaining_three_with_chart.py
├── cross_dataset_generalization.py
├── gradcam_colab_exact.py
├── requirements.txt
└── README.md
```

### Main Scripts

| Script                                         | Description                                                          |
| ---------------------------------------------- | -------------------------------------------------------------------- |
| `experiment_cnn_transformer_full_pipeline.py`  | Training and evaluation of the proposed hybrid CNN–Transformer model |
| `ablation_study_remaining_three_with_chart.py` | Reproduces the ablation study                                        |
| `cross_dataset_generalization.py`              | Performs external validation on an independent MRI dataset           |
| `gradcam_colab_exact.py`                       | Generates Grad-CAM visualizations                                    |

---

## Datasets

### 1. Brain Tumor MRI Dataset (Training and Internal Evaluation)

**Author:** Mohammad Nickparvar

**Kaggle:**

https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

**DOI:**

10.34740/KAGGLE/DSV/2645886

The dataset contains approximately 7,000 MRI images distributed across four classes:

* Glioma
* Meningioma
* Pituitary Tumor
* No Tumor

---

### 2. Brain Tumor MRI Dataset (External Validation)

**Authors:**

* Md Irfanul Kabir Hira
* Md Sohag Hossain
* Mst Moriom Akter Bithee

**Mendeley Data:**

https://data.mendeley.com/datasets/zwr4ntf94j/2

**DOI:**

10.17632/zwr4ntf94j.2

The dataset contains 12,064 MRI images and was used exclusively for external validation.

No retraining, fine-tuning, or domain adaptation was performed before external evaluation.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/brain-tumor-cnn-transformer.git
cd brain-tumor-cnn-transformer
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Dataset Preparation

### Kaggle Dataset Structure

```text
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
```

### External Dataset Structure

```text
External_Mendeley/
├── glioma/
├── meningioma/
├── pituitary/
└── notumor/
```

---

## Reproducing the Experiments

### Train the Proposed Hybrid CNN–Transformer Model

```bash
python experiment_cnn_transformer_full_pipeline.py
```

Outputs:

* Trained model
* Accuracy and loss curves
* Confusion matrix
* Classification report
* Evaluation metrics

---

### Run the Ablation Study

```bash
python ablation_study_remaining_three_with_chart.py
```

This reproduces:

* Transformer-only model
* CNN-only model
* Hybrid without positional embeddings
* Full hybrid CNN–Transformer model

Outputs:

* Ablation metrics
* Performance comparison chart

---

### Generate Grad-CAM Visualizations

```bash
python gradcam_colab_exact.py
```

Outputs:

* Grad-CAM heatmaps
* Overlay visualizations
* Explainability figures

---

### Perform External Validation

```bash
python cross_dataset_generalization.py \
    --model_path best_model.keras \
    --class_indices_path class_indices.json \
    --external_data_root path/to/External_Mendeley \
    --output_dir external_results
```

Outputs:

* External confusion matrix
* External classification report
* Cross-dataset evaluation metrics

---

## Model Configuration

| Parameter              | Value               |
| ---------------------- | ------------------- |
| Input Size             | 224 × 224 × 3       |
| CNN Blocks             | 4                   |
| CNN Channels           | 32 / 64 / 128 / 256 |
| Token Count            | 196                 |
| Embedding Dimension    | 128                 |
| Transformer Blocks     | 2                   |
| Attention Heads        | 2                   |
| Feed-Forward Dimension | 256                 |
| Dropout                | 0.25                |
| Label Smoothing        | 0.08                |
| Optimizer              | Adam                |
| Learning Rate          | 1e-4                |

---

## Environment

Experiments were conducted using:

* Python 3
* TensorFlow 2.x
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* OpenCV
* Google Colab
* NVIDIA Tesla T4 GPU

---

## Reproducibility Notes

* The same preprocessing pipeline was used across all experiments.
* External validation was performed without retraining or fine-tuning.
* All experiments were originally executed in Google Colab.
* Results may vary slightly across hardware and TensorFlow versions due to nondeterministic GPU operations.
