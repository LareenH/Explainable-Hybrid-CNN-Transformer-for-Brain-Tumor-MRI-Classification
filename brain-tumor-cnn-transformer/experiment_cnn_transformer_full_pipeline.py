
"""
Full Pure CNN + Transformer Experiment Pipeline
==============================================

Brain Tumor MRI Classification - CNN + Transformer from scratch.
No transfer learning. No ResNet/EfficientNet.

Included:
- Dataset auto-detection:
    1) data_root/train/<class> and data_root/test/<class>
    2) data_root/Training/<class> and data_root/Testing/<class>
    3) data_root/<class> only -> automatic train/val/test split
- Preprocessing:
    - image resizing
    - RGB conversion through Keras generator
    - normalization using Rescaling(1/255) inside model
- Augmentation:
    - rotation, shift, zoom, shear, brightness, horizontal flip
- Regularization:
    - BatchNormalization
    - Dropout
    - L2 weight decay
    - Label smoothing
    - class weights
    - EarlyStopping
    - ReduceLROnPlateau
- Outputs:
    - best model
    - final model
    - model summary
    - class indices
    - training curves
    - classification reports
    - confusion matrices
    - prediction CSV
    - model_results.csv

Example:
    source /home/lhaddad/Desktop/ML/brain_env/bin/activate

    python3 experiment_cnn_transformer_full_pipeline.py \
      --data_root "/home/lhaddad/Desktop/newdata" \
      --epochs 60 \
      --batch_size 8 \
      --output_dir "results_7k_cnn_transformer"
"""

import argparse
import json
import os
import random
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras import Model, layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def set_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def find_split_dirs(data_root: Path):
    candidates = [
        ("Training", "Testing"),
        ("training", "testing"),
        ("train", "test"),
        ("Train", "Test"),
    ]

    for train_name, test_name in candidates:
        train_dir = data_root / train_name
        test_dir = data_root / test_name
        if train_dir.exists() and test_dir.exists():
            return train_dir, test_dir

    return None, None


def list_class_dirs(folder: Path):
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.is_dir()])


def has_class_subdirs(folder: Path):
    class_dirs = list_class_dirs(folder)
    if len(class_dirs) < 2:
        return False

    for class_dir in class_dirs:
        if any(f.suffix.lower() in IMAGE_EXTENSIONS for f in class_dir.rglob("*")):
            return True

    return False


def create_split_from_single_folder(data_root: Path, output_dir: Path, seed: int, val_ratio=0.15, test_ratio=0.15):
    split_root = output_dir / "split_data"

    if (split_root / "train").exists() and (split_root / "val").exists() and (split_root / "test").exists():
        print(f"Using existing generated split: {split_root}")
        return split_root / "train", split_root / "val", split_root / "test"

    print("No train/test folders found. Creating train/val/test split...")
    split_root.mkdir(parents=True, exist_ok=True)

    class_dirs = list_class_dirs(data_root)
    if len(class_dirs) < 2:
        raise ValueError(
            f"Could not detect class folders in {data_root}. "
            "Expected either train/test folders or class subfolders."
        )

    for split in ["train", "val", "test"]:
        for class_dir in class_dirs:
            (split_root / split / class_dir.name).mkdir(parents=True, exist_ok=True)

    for class_dir in class_dirs:
        files = [
            p for p in class_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]

        if len(files) < 5:
            print(f"Warning: class {class_dir.name} has only {len(files)} images.")

        train_val_files, test_files = train_test_split(
            files,
            test_size=test_ratio,
            random_state=seed,
            shuffle=True,
        )

        adjusted_val_ratio = val_ratio / (1.0 - test_ratio)

        train_files, val_files = train_test_split(
            train_val_files,
            test_size=adjusted_val_ratio,
            random_state=seed,
            shuffle=True,
        )

        for split_name, split_files in [
            ("train", train_files),
            ("val", val_files),
            ("test", test_files),
        ]:
            for src in split_files:
                dst = split_root / split_name / class_dir.name / src.name
                if not dst.exists():
                    shutil.copy2(src, dst)

        print(
            f"{class_dir.name}: train={len(train_files)}, "
            f"val={len(val_files)}, test={len(test_files)}"
        )

    return split_root / "train", split_root / "val", split_root / "test"


def prepare_dataset(data_root: str, output_dir: str, seed: int):
    data_root = Path(data_root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_root.exists():
        raise FileNotFoundError(f"data_root does not exist: {data_root}")

    train_dir, test_dir = find_split_dirs(data_root)

    if train_dir is not None:
        # If train/test exist, create validation from train using validation_split in generator.
        print(f"Detected train directory: {train_dir}")
        print(f"Detected test directory:  {test_dir}")
        return train_dir, None, test_dir, "split_with_train_test"

    if has_class_subdirs(data_root):
        train_dir, val_dir, test_dir = create_split_from_single_folder(
            data_root=data_root,
            output_dir=output_dir,
            seed=seed,
        )
        return train_dir, val_dir, test_dir, "generated_train_val_test"

    raise ValueError(
        "Unsupported dataset structure. Expected train/test folders, "
        "Training/Testing folders, or class folders directly under data_root."
    )


def make_generators(train_dir, val_dir, test_dir, img_size, batch_size, seed):
    """
    Uses augmentation only for training.
    Validation/test are not augmented.
    Normalization is done inside the model with Rescaling(1/255).
    Therefore ImageDataGenerator does not rescale here.
    """

    train_aug = ImageDataGenerator(
        validation_split=0.2 if val_dir is None else 0.0,
        rotation_range=15,
        width_shift_range=0.08,
        height_shift_range=0.08,
        zoom_range=0.12,
        shear_range=0.05,
        brightness_range=(0.85, 1.15),
        horizontal_flip=True,
        fill_mode="nearest",
    )

    eval_aug = ImageDataGenerator(
        validation_split=0.2 if val_dir is None else 0.0,
    )

    if val_dir is None:
        train_gen = train_aug.flow_from_directory(
            train_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode="categorical",
            subset="training",
            shuffle=True,
            seed=seed,
        )

        val_gen = eval_aug.flow_from_directory(
            train_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode="categorical",
            subset="validation",
            shuffle=False,
            seed=seed,
        )
    else:
        train_gen = train_aug.flow_from_directory(
            train_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=True,
            seed=seed,
        )

        val_gen = eval_aug.flow_from_directory(
            val_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=False,
        )

    test_gen = ImageDataGenerator().flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    if train_gen.class_indices != test_gen.class_indices:
        print("Warning: train/test class mapping mismatch.")
        print("Train:", train_gen.class_indices)
        print("Test: ", test_gen.class_indices)

    return train_gen, val_gen, test_gen


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.15, l2_reg=1e-4, **kwargs):
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
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        x = self.layernorm1(inputs + attn_output)

        ffn_output = self.ffn(x, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(x + ffn_output)


def conv_block(x, filters, dropout=0.0, l2_reg=1e-4):
    x = layers.Conv2D(
        filters,
        kernel_size=3,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(l2_reg),
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(
        filters,
        kernel_size=3,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(l2_reg),
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.MaxPooling2D(pool_size=2)(x)

    if dropout > 0:
        x = layers.Dropout(dropout)(x)

    return x


def build_cnn_transformer(
    img_size=(224, 224),
    num_classes=4,
    embed_dim=128,
    num_heads=2,
    transformer_depth=2,
    ff_dim=256,
    dropout=0.25,
    l2_reg=1e-4,
):
    inputs = layers.Input(shape=(img_size[0], img_size[1], 3))

    # Normalization/preprocessing
    x = layers.Rescaling(1.0 / 255.0, name="normalization_rescale")(inputs)

    # CNN feature extractor from scratch
    x = conv_block(x, 32, dropout=0.05, l2_reg=l2_reg)
    x = conv_block(x, 64, dropout=0.10, l2_reg=l2_reg)
    x = conv_block(x, 128, dropout=0.15, l2_reg=l2_reg)
    x = conv_block(x, 256, dropout=0.20, l2_reg=l2_reg)

    # 224 -> 112 -> 56 -> 28 -> 14, so tokens = 14*14 = 196
    x = layers.Conv2D(embed_dim, kernel_size=1, padding="same", name="token_projection")(x)

    h = x.shape[1]
    w = x.shape[2]
    if h is None or w is None:
        raise ValueError("Image size must be fixed so token count can be computed.")

    num_tokens = int(h * w)

    x = layers.Reshape((num_tokens, embed_dim), name="spatial_tokens")(x)

    # Positional embedding with correct broadcasting: (1, tokens, embed_dim)
    positions = tf.range(start=0, limit=num_tokens, delta=1)
    position_embedding = layers.Embedding(
        input_dim=num_tokens,
        output_dim=embed_dim,
        name="positional_embedding",
    )(positions)
    position_embedding = tf.expand_dims(position_embedding, axis=0)
    x = layers.Add(name="add_position")([x, position_embedding])

    for i in range(transformer_depth):
        x = TransformerBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            l2_reg=l2_reg,
            name=f"transformer_block_{i+1}",
        )(x)

    x = layers.LayerNormalization(epsilon=1e-6, name="transformer_output_norm")(x)
    x = layers.GlobalAveragePooling1D(name="token_pooling")(x)

    x = layers.Dropout(0.40)(x)
    x = layers.Dense(
        256,
        activation="gelu",
        kernel_regularizer=regularizers.l2(l2_reg),
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.35)(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="classifier")(x)

    return Model(inputs=inputs, outputs=outputs, name="Pure_CNN_Transformer")


def compute_class_weights(generator):
    classes = generator.classes
    labels = np.unique(classes)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=labels,
        y=classes,
    )
    return {int(label): float(weight) for label, weight in zip(labels, weights)}


def plot_training_history(history, output_dir):
    output_dir = Path(output_dir)
    hist = pd.DataFrame(history.history)
    hist.to_csv(output_dir / "training_history.csv", index=False)

    for metric in ["accuracy", "loss"]:
        plt.figure(figsize=(8, 5))
        plt.plot(hist[metric], label=f"train_{metric}")
        val_metric = f"val_{metric}"
        if val_metric in hist:
            plt.plot(hist[val_metric], label=val_metric)
        plt.title(f"Training and Validation {metric.capitalize()}")
        plt.xlabel("Epoch")
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f"{metric}_curve.png", dpi=300)
        plt.close()

    if "learning_rate" in hist.columns:
        lr_col = "learning_rate"
    elif "lr" in hist.columns:
        lr_col = "lr"
    else:
        lr_col = None

    if lr_col:
        plt.figure(figsize=(8, 5))
        plt.plot(hist[lr_col], label="learning_rate")
        plt.title("Learning Rate Schedule")
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "learning_rate_curve.png", dpi=300)
        plt.close()


def evaluate_model(model, generator, class_names, output_dir, model_name):
    output_dir = Path(output_dir)
    generator.reset()

    y_prob = model.predict(generator, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = generator.classes

    filenames = generator.filenames

    acc = accuracy_score(y_true, y_pred)

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
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

    report_txt = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
    )
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    with open(output_dir / f"{model_name}_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report_txt)

    pd.DataFrame(report_dict).T.to_csv(output_dir / f"{model_name}_classification_report.csv")

    cm = confusion_matrix(y_true, y_pred)
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
        output_dir / f"{model_name}_confusion_matrix.csv"
    )

    plt.figure(figsize=(8, 7))
    plt.imshow(cm)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(range(len(class_names)), class_names)
    plt.colorbar()

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name}_confusion_matrix.png", dpi=300)
    plt.close()

    pred_df = pd.DataFrame({
        "filename": filenames,
        "true_label_index": y_true,
        "predicted_label_index": y_pred,
        "true_label": [class_names[i] for i in y_true],
        "predicted_label": [class_names[i] for i in y_pred],
        "correct": y_true == y_pred,
    })

    for i, class_name in enumerate(class_names):
        pred_df[f"prob_{class_name}"] = y_prob[:, i]

    pred_df.to_csv(output_dir / f"{model_name}_predictions.csv", index=False)

    return {
        "model": model_name,
        "accuracy": acc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "macro_auc_ovr": auc_macro_ovr,
    }


def save_sample_batch(generator, output_dir, class_names):
    output_dir = Path(output_dir)
    images, labels = next(generator)
    count = min(12, len(images))

    plt.figure(figsize=(12, 9))
    for i in range(count):
        plt.subplot(3, 4, i + 1)
        img = images[i].astype("uint8")
        plt.imshow(img)
        label_idx = int(np.argmax(labels[i]))
        plt.title(class_names[label_idx])
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_dir / "sample_augmented_training_images.png", dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_root",
        default="/home/lhaddad/Desktop/newdata",
        help="Dataset folder. Supports train/test, Training/Testing, or class folders.",
    )
    parser.add_argument("--output_dir", default="results_7k_cnn_transformer")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--transformer_depth", type=int, default=2)
    parser.add_argument("--ff_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--l2_reg", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.08)
    parser.add_argument("--learning_rate", type=float, default=1e-4)

    args = parser.parse_args()

    set_seed(args.seed)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dir, val_dir, test_dir, dataset_mode = prepare_dataset(
        args.data_root,
        output_dir,
        args.seed,
    )

    img_size = (args.img_size, args.img_size)

    train_gen, val_gen, test_gen = make_generators(
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir,
        img_size=img_size,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    class_names = list(train_gen.class_indices.keys())
    num_classes = len(class_names)

    with open(output_dir / "class_indices.json", "w", encoding="utf-8") as f:
        json.dump(train_gen.class_indices, f, indent=2)

    with open(output_dir / "experiment_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args) | {
            "dataset_mode": dataset_mode,
            "train_dir": str(train_dir),
            "val_dir": str(val_dir) if val_dir else None,
            "test_dir": str(test_dir),
            "class_names": class_names,
        }, f, indent=2)

    save_sample_batch(train_gen, output_dir, class_names)

    class_weights = compute_class_weights(train_gen)
    print("Class weights:", class_weights)

    model = build_cnn_transformer(
        img_size=img_size,
        num_classes=num_classes,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        transformer_depth=args.transformer_depth,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        l2_reg=args.l2_reg,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing),
        metrics=["accuracy"],
    )

    model.summary()

    with open(output_dir / "model_summary.txt", "w", encoding="utf-8") as f:
        model.summary(print_fn=lambda line: f.write(line + "\n"))

    callbacks = [
        ModelCheckpoint(
            output_dir / "best_model.keras",
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=12,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    plot_training_history(history, output_dir)

    model.save(output_dir / "final_model.keras")

    results = []
    results.append(evaluate_model(model, val_gen, class_names, output_dir, "cnn_transformer_val"))
    results.append(evaluate_model(model, test_gen, class_names, output_dir, "cnn_transformer_test"))

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "model_results.csv", index=False)

    print("\nSaved results to:", output_dir)
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
