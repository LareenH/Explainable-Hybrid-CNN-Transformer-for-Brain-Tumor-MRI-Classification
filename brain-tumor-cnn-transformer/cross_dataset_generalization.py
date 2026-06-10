
"""
Cross-Dataset Generalization Evaluation
=======================================

Purpose:
    Evaluate the trained hybrid CNN-Transformer model on an external dataset
    WITHOUT retraining.

Use case:
    Train on the 7k Kaggle dataset, then test directly on another dataset
    such as the older 3k Brain-Tumor-Classification dataset.

This script:
- Loads your trained .keras model.
- Handles the custom TransformerBlock.
- Loads class order from class_indices.json if available.
- Reads external dataset folders.
- Maps common folder names such as:
    glioma_tumor -> glioma
    meningioma_tumor -> meningioma
    pituitary_tumor -> pituitary
    no_tumor -> notumor
    notumor -> notumor
- Computes:
    accuracy
    macro precision / recall / F1
    weighted precision / recall / F1
    macro AUC OVR
    classification report
    confusion matrix
    predictions CSV
- Saves output visualizations and CSV files.

Example local run:

    source ~/Desktop/ML/brain_env/bin/activate

    python cross_dataset_generalization.py \
      --model_path "/home/lhaddad/Desktop/newdata/results_7k_cnn_transformer/best_model.keras" \
      --class_indices_path "/home/lhaddad/Desktop/newdata/results_7k_cnn_transformer/class_indices.json" \
      --external_data_root "/home/lhaddad/brain_tumor_project/data/Brain-Tumor-Classification-DataSet-master" \
      --output_dir "/home/lhaddad/Desktop/newdata/results_cross_dataset_3k"

Example Colab run:

    !python3 /content/cross_dataset_generalization.py \
      --model_path "/content/drive/MyDrive/MRI_Ablation/results_hybrid_with_position_colab/best_model.keras" \
      --class_indices_path "/content/drive/MyDrive/MRI_Ablation/results_hybrid_with_position_colab/class_indices.json" \
      --external_data_root "/content/external_dataset" \
      --output_dir "/content/drive/MyDrive/MRI_Ablation/results_cross_dataset_external"
"""

import argparse
import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from tensorflow.keras import layers, regularizers


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# Exact custom layer from your training script.
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.25, l2_reg=1e-4, **kwargs):
        super().__init__(**kwargs)

        self.att = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=dropout,
        )

        self.ffn = tf.keras.Sequential([
            layers.Dense(
                ff_dim,
                activation="gelu",
                kernel_regularizer=regularizers.l2(l2_reg),
            ),
            layers.Dropout(dropout),
            layers.Dense(
                embed_dim,
                kernel_regularizer=regularizers.l2(l2_reg),
            ),
        ])

        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)

        self.drop1 = layers.Dropout(dropout)
        self.drop2 = layers.Dropout(dropout)

    def call(self, inputs, training=False):
        attn = self.att(inputs, inputs, training=training)
        attn = self.drop1(attn, training=training)

        x = self.norm1(inputs + attn)

        ffn = self.ffn(x, training=training)
        ffn = self.drop2(ffn, training=training)

        return self.norm2(x + ffn)


def normalize_label(name):
    """
    Converts common external dataset folder names into the 7k model class names.
    """
    s = name.lower().strip()
    s = s.replace("-", "_").replace(" ", "_")

    mapping = {
        "glioma": "glioma",
        "glioma_tumor": "glioma",
        "glioma tumor": "glioma",

        "meningioma": "meningioma",
        "meningioma_tumor": "meningioma",
        "meningioma tumor": "meningioma",

        "pituitary": "pituitary",
        "pituitary_tumor": "pituitary",
        "pituitary tumor": "pituitary",

        "notumor": "notumor",
        "no_tumor": "notumor",
        "no tumor": "notumor",
        "no_tumour": "notumor",
        "notumour": "notumor",
        "normal": "notumor",
    }

    return mapping.get(s, s)


def load_class_names(class_indices_path, model_path):
    possible_paths = []

    if class_indices_path:
        possible_paths.append(Path(class_indices_path))

    model_dir = Path(model_path).parent
    possible_paths.extend([
        model_dir / "class_indices.json",
        model_dir / "experiment_config.json",
    ])

    for path in possible_paths:
        if not path.exists():
            continue

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "class_indices" in data:
            class_indices = data["class_indices"]
            class_names = [None] * len(class_indices)
            for name, idx in class_indices.items():
                class_names[int(idx)] = name
            return class_names

        if "class_names" in data:
            return data["class_names"]

        # direct class_indices.json usually has {"glioma":0, ...}
        if all(isinstance(v, int) for v in data.values()):
            class_names = [None] * len(data)
            for name, idx in data.items():
                class_names[int(idx)] = name
            return class_names

    raise FileNotFoundError(
        "Could not load class names. Please provide --class_indices_path."
    )


def find_external_eval_root(external_data_root):
    """
    Accepts:
        root/test/<class>
        root/Testing/<class>
        root/<class>
    Returns the folder containing class folders.
    """
    root = Path(external_data_root).expanduser().resolve()

    if not root.exists():
        raise FileNotFoundError(f"external_data_root does not exist: {root}")

    candidates = [
        root / "Testing",
        root / "testing",
        root / "Test",
        root / "test",
        root,
    ]

    for candidate in candidates:
        if not candidate.exists():
            continue

        class_dirs = [p for p in candidate.iterdir() if p.is_dir()]
        if len(class_dirs) >= 2:
            has_images = False
            for class_dir in class_dirs:
                if any(f.suffix.lower() in IMAGE_EXTENSIONS for f in class_dir.rglob("*")):
                    has_images = True
                    break
            if has_images:
                return candidate

    raise ValueError(
        f"Could not find class folders under external dataset root: {root}"
    )


def collect_external_images(eval_root, model_class_names):
    rows = []
    model_class_set = set(model_class_names)

    for class_dir in sorted([p for p in Path(eval_root).iterdir() if p.is_dir()]):
        raw_label = class_dir.name
        mapped_label = normalize_label(raw_label)

        if mapped_label not in model_class_set:
            print(
                f"Warning: skipping folder '{raw_label}' mapped to '{mapped_label}' "
                f"because it is not in model classes {model_class_names}"
            )
            continue

        for path in class_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                rows.append({
                    "filepath": str(path),
                    "raw_label": raw_label,
                    "mapped_label": mapped_label,
                    "true_index": model_class_names.index(mapped_label),
                })

    if not rows:
        raise ValueError(
            "No usable images found after class-name mapping. "
            "Check folder names and model class names."
        )

    return pd.DataFrame(rows)


def load_image_batch(paths, img_size):
    batch = []
    for path in paths:
        img = tf.keras.utils.load_img(path, target_size=(img_size, img_size))
        arr = tf.keras.utils.img_to_array(img)
        batch.append(arr)

    return np.asarray(batch, dtype=np.float32)


def predict_external(model, df, img_size, batch_size):
    probs = []

    paths = df["filepath"].tolist()

    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start:start + batch_size]
        x = load_image_batch(batch_paths, img_size)
        y = model.predict(x, verbose=0)
        probs.append(y)

        done = min(start + batch_size, len(paths))
        print(f"Predicted {done}/{len(paths)} images")

    return np.vstack(probs)


def save_confusion_matrix(cm, class_names, output_path, title):
    plt.figure(figsize=(8, 7))
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(range(len(class_names)), class_names)
    plt.colorbar()

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", required=True)
    parser.add_argument("--class_indices_path", default=None)
    parser.add_argument("--external_data_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)

    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model:", args.model_path)
    model = tf.keras.models.load_model(
        args.model_path,
        compile=False,
        custom_objects={"TransformerBlock": TransformerBlock},
        safe_mode=False,
    )

    class_names = load_class_names(args.class_indices_path, args.model_path)
    print("Model class names:", class_names)

    eval_root = find_external_eval_root(args.external_data_root)
    print("External evaluation folder:", eval_root)

    df = collect_external_images(eval_root, class_names)
    print("\nExternal dataset distribution after mapping:")
    print(df["mapped_label"].value_counts().to_string())

    y_true = df["true_index"].values

    y_prob = predict_external(
        model=model,
        df=df,
        img_size=args.img_size,
        batch_size=args.batch_size,
    )

    y_pred = np.argmax(y_prob, axis=1)

    df["predicted_index"] = y_pred
    df["predicted_label"] = [class_names[i] for i in y_pred]
    df["correct"] = y_pred == y_true

    for i, class_name in enumerate(class_names):
        df[f"prob_{class_name}"] = y_prob[:, i]

    acc = accuracy_score(y_true, y_pred)

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    try:
        y_true_onehot = tf.keras.utils.to_categorical(y_true, num_classes=len(class_names))
        auc_macro_ovr = roc_auc_score(
            y_true_onehot,
            y_prob,
            multi_class="ovr",
            average="macro",
        )
    except Exception:
        auc_macro_ovr = np.nan

    results = {
        "model": "cross_dataset_external_test",
        "external_data_root": str(Path(args.external_data_root).expanduser().resolve()),
        "external_eval_root": str(eval_root),
        "num_images": int(len(df)),
        "accuracy": acc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "macro_auc_ovr": auc_macro_ovr,
    }

    pd.DataFrame([results]).to_csv(output_dir / "cross_dataset_results.csv", index=False)

    report_txt = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
    )

    with open(output_dir / "cross_dataset_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report_txt)

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report_dict).T.to_csv(output_dir / "cross_dataset_classification_report.csv")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
        output_dir / "cross_dataset_confusion_matrix.csv"
    )

    save_confusion_matrix(
        cm=cm,
        class_names=class_names,
        output_path=output_dir / "cross_dataset_confusion_matrix.png",
        title="Cross-Dataset Confusion Matrix",
    )

    df.to_csv(output_dir / "cross_dataset_predictions.csv", index=False)

    with open(output_dir / "cross_dataset_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print("\nSaved cross-dataset results to:", output_dir)
    print(pd.DataFrame([results]).to_string(index=False))


if __name__ == "__main__":
    main()
