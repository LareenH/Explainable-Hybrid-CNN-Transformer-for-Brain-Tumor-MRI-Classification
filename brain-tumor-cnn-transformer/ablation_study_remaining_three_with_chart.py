
"""
Ablation Study - Remaining Three Models Only
============================================

This script trains ONLY the remaining ablation models:

1. CNN-only
2. Transformer-only
3. Hybrid CNN+Transformer WITHOUT positional embedding

It does NOT train the full hybrid model with positional embedding because that result
already exists from the main experiment:

Hybrid CNN+Transformer WITH positional embedding:
- Test Accuracy: 0.981119
- Macro F1: 0.981035
- Macro AUC: 0.997548

Fair-comparison settings matched to the successful full pipeline:
- Dataset path: configurable by --data_root
- Train/test detection: train/test or Training/Testing
- Image size: 224
- Batch size: 8
- Epoch budget: 60
- Validation split: 0.20
- Augmentation: same ImageDataGenerator setup as full pipeline
- Normalization: Rescaling(1/255) inside the model
- Label smoothing: 0.08
- L2 regularization: 1e-4
- Adam learning rate: 1e-4
- Class weights: enabled
- Early stopping: val_loss, patience 12
- ReduceLROnPlateau: val_loss, patience 5, min_lr 1e-7
- ModelCheckpoint: val_accuracy

Run:
    source /home/lhaddad/Desktop/ML/brain_env/bin/activate

    python3 ablation_study_remaining_three_with_chart.py \
      --data_root "/home/lhaddad/Desktop/newdata" \
      --output_dir "/home/lhaddad/Desktop/newdata/results_ablation_remaining_three" \
      --epochs 60 \
      --batch_size 8
"""

import argparse
import json
import os
import random
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
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras import Model, layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# -----------------------------
# Data
# -----------------------------
def find_split_dirs(data_root):
    data_root = Path(data_root)

    for train_name, test_name in [
        ("Training", "Testing"),
        ("training", "testing"),
        ("Train", "Test"),
        ("train", "test"),
    ]:
        train_dir = data_root / train_name
        test_dir = data_root / test_name
        if train_dir.exists() and test_dir.exists():
            return train_dir, test_dir

    raise FileNotFoundError(
        f"Could not find train/test folders under {data_root}. "
        "Expected Training/Testing or train/test."
    )


def make_generators(data_root, img_size, batch_size, seed):
    train_dir, test_dir = find_split_dirs(data_root)

    # Same as the successful full pipeline.
    # Important: no rescale here because normalization is inside each model.
    train_aug = ImageDataGenerator(
        validation_split=0.2,
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
        validation_split=0.2,
    )

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

    test_gen = ImageDataGenerator().flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    return train_gen, val_gen, test_gen, train_dir, test_dir


def compute_weights(generator):
    labels = generator.classes
    classes = np.unique(labels)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels,
    )

    return {int(c): float(w) for c, w in zip(classes, weights)}


# -----------------------------
# Transformer block
# -----------------------------
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


# -----------------------------
# Shared model components
# -----------------------------
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


def cnn_feature_extractor(inputs, l2_reg=1e-4):
    # Same normalization location as full experiment.
    x = layers.Rescaling(1.0 / 255.0, name="normalization_rescale")(inputs)

    # Same CNN blocks and dropout as full experiment.
    x = conv_block(x, 32, dropout=0.05, l2_reg=l2_reg)
    x = conv_block(x, 64, dropout=0.10, l2_reg=l2_reg)
    x = conv_block(x, 128, dropout=0.15, l2_reg=l2_reg)
    x = conv_block(x, 256, dropout=0.20, l2_reg=l2_reg)

    return x


def classifier_head(x, num_classes, l2_reg=1e-4):
    # Same classifier head as full experiment.
    x = layers.Dropout(0.40)(x)
    x = layers.Dense(
        256,
        activation="gelu",
        kernel_regularizer=regularizers.l2(l2_reg),
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.35)(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="classifier")(x)

    return outputs


# -----------------------------
# Ablation models
# -----------------------------
def build_cnn_only(img_size, num_classes, l2_reg=1e-4):
    inputs = layers.Input(shape=(img_size[0], img_size[1], 3))

    x = cnn_feature_extractor(inputs, l2_reg=l2_reg)

    # Ablation: replace Transformer blocks with standard Global Average Pooling.
    x = layers.GlobalAveragePooling2D(name="cnn_gap")(x)

    outputs = classifier_head(x, num_classes, l2_reg=l2_reg)

    return Model(inputs, outputs, name="cnn_only")


def add_positional_embedding(x, num_tokens, embed_dim):
    positions = tf.range(start=0, limit=num_tokens, delta=1)

    pos = layers.Embedding(
        input_dim=num_tokens,
        output_dim=embed_dim,
        name="positional_embedding",
    )(positions)

    pos = tf.expand_dims(pos, axis=0)

    return layers.Add(name="add_position")([x, pos])


def build_transformer_only(
    img_size,
    num_classes,
    patch_size=16,
    embed_dim=128,
    num_heads=2,
    depth=2,
    ff_dim=256,
    dropout=0.25,
    l2_reg=1e-4,
):
    inputs = layers.Input(shape=(img_size[0], img_size[1], 3))

    # Same normalization location.
    x = layers.Rescaling(1.0 / 255.0, name="normalization_rescale")(inputs)

    # Ablation: no CNN blocks. Raw image is divided into non-overlapping patches.
    x = layers.Conv2D(
        embed_dim,
        kernel_size=patch_size,
        strides=patch_size,
        padding="valid",
        name="patch_embedding",
        kernel_regularizer=regularizers.l2(l2_reg),
    )(x)

    h, w = int(x.shape[1]), int(x.shape[2])
    num_tokens = h * w

    x = layers.Reshape((num_tokens, embed_dim), name="image_patches")(x)

    # Transformer-only still needs position, as in standard ViT-style patch models.
    x = add_positional_embedding(x, num_tokens, embed_dim)

    for i in range(depth):
        x = TransformerBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            l2_reg=l2_reg,
            name=f"transformer_block_{i+1}",
        )(x)

    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.GlobalAveragePooling1D(name="patch_pooling")(x)

    outputs = classifier_head(x, num_classes, l2_reg=l2_reg)

    return Model(inputs, outputs, name="transformer_only")


def build_hybrid_no_position(
    img_size,
    num_classes,
    embed_dim=128,
    num_heads=2,
    depth=2,
    ff_dim=256,
    dropout=0.25,
    l2_reg=1e-4,
):
    inputs = layers.Input(shape=(img_size[0], img_size[1], 3))

    x = cnn_feature_extractor(inputs, l2_reg=l2_reg)

    x = layers.Conv2D(embed_dim, kernel_size=1, padding="same", name="token_projection")(x)

    h, w = int(x.shape[1]), int(x.shape[2])
    num_tokens = h * w

    x = layers.Reshape((num_tokens, embed_dim), name="spatial_tokens")(x)

    # Important ablation:
    # No positional embedding and no add_position layer here.

    for i in range(depth):
        x = TransformerBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            l2_reg=l2_reg,
            name=f"transformer_block_{i+1}",
        )(x)

    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.GlobalAveragePooling1D(name="token_pooling")(x)

    outputs = classifier_head(x, num_classes, l2_reg=l2_reg)

    return Model(inputs, outputs, name="hybrid_no_position")


# -----------------------------
# Training/evaluation utilities
# -----------------------------
def compile_model(model, learning_rate, label_smoothing):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        metrics=["accuracy"],
    )


def evaluate(model, generator, class_names, output_dir, model_name):
    generator.reset()

    y_prob = model.predict(generator, verbose=1)

    y_pred = np.argmax(y_prob, axis=1)
    y_true = generator.classes

    acc = accuracy_score(y_true, y_pred)

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    try:
        y_true_hot = tf.keras.utils.to_categorical(y_true, num_classes=len(class_names))
        auc = roc_auc_score(
            y_true_hot,
            y_prob,
            multi_class="ovr",
            average="macro",
        )
    except Exception:
        auc = np.nan

    report_txt = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
    )

    with open(Path(output_dir) / f"{model_name}_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report_txt)

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report_dict).T.to_csv(Path(output_dir) / f"{model_name}_classification_report.csv")

    cm = confusion_matrix(y_true, y_pred)

    pd.DataFrame(
        cm,
        index=class_names,
        columns=class_names,
    ).to_csv(Path(output_dir) / f"{model_name}_confusion_matrix.csv")

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
    plt.savefig(Path(output_dir) / f"{model_name}_confusion_matrix.png", dpi=300)
    plt.close()

    return {
        "model": model_name,
        "accuracy": acc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "macro_auc_ovr": auc,
    }


def plot_history(history, output_dir, model_name):
    hist = pd.DataFrame(history.history)

    hist.to_csv(Path(output_dir) / f"{model_name}_history.csv", index=False)

    for metric in ["accuracy", "loss"]:
        plt.figure(figsize=(8, 5))
        plt.plot(hist[metric], label=f"train_{metric}")

        if f"val_{metric}" in hist:
            plt.plot(hist[f"val_{metric}"], label=f"val_{metric}")

        plt.title(f"{model_name} {metric}")
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(Path(output_dir) / f"{model_name}_{metric}.png", dpi=300)
        plt.close()


def plot_ablation_chart(results_df, output_dir, filename="ablation_comparison_remaining_three.png"):
    test_df = results_df[results_df["model"].str.endswith("_test")].copy()

    # Clean model names for plotting.
    test_df["display_model"] = (
        test_df["model"]
        .str.replace("_test", "", regex=False)
        .str.replace("_", " ")
        .str.title()
    )

    x = np.arange(len(test_df))
    width = 0.35

    plt.figure(figsize=(10, 6))

    plt.bar(x - width / 2, test_df["accuracy"], width, label="Accuracy")
    plt.bar(x + width / 2, test_df["macro_f1"], width, label="Macro F1")

    plt.xticks(x, test_df["display_model"], rotation=15, ha="right")
    plt.ylabel("Score")
    plt.title("Ablation Study - Remaining Three Models")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(axis="y", alpha=0.25)

    for i, value in enumerate(test_df["accuracy"]):
        plt.text(i - width / 2, value + 0.01, f"{value:.4f}", ha="center", fontsize=8)

    for i, value in enumerate(test_df["macro_f1"]):
        plt.text(i + width / 2, value + 0.01, f"{value:.4f}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(Path(output_dir) / filename, dpi=300)
    plt.close()


def run_single_model(model_name, model, train_gen, val_gen, test_gen, class_names, class_weights, args):
    model_dir = Path(args.output_dir) / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    compile_model(
        model,
        learning_rate=args.learning_rate,
        label_smoothing=args.label_smoothing,
    )

    with open(model_dir / "model_summary.txt", "w", encoding="utf-8") as f:
        model.summary(print_fn=lambda line: f.write(line + "\n"))

    callbacks = [
        ModelCheckpoint(
            model_dir / "best_model.keras",
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=args.lr_patience,
            min_lr=args.min_lr,
            verbose=1,
        ),
    ]

    print("\n" + "=" * 80)
    print("Training:", model_name)
    print("=" * 80)

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    plot_history(history, model_dir, model_name)

    model.save(model_dir / "final_model.keras")

    val_metrics = evaluate(model, val_gen, class_names, model_dir, f"{model_name}_val")
    test_metrics = evaluate(model, test_gen, class_names, model_dir, f"{model_name}_test")

    return val_metrics, test_metrics


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", default="/home/lhaddad/Desktop/newdata")
    parser.add_argument("--output_dir", default="/home/lhaddad/Desktop/newdata/results_ablation_remaining_three")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--lr_patience", type=int, default=5)
    parser.add_argument("--min_lr", type=float, default=1e-7)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.08)

    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--ff_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--l2_reg", type=float, default=1e-4)
    parser.add_argument("--patch_size", type=int, default=16)

    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "cnn_only",
            "transformer_only",
            "hybrid_no_position",
        ],
        choices=[
            "cnn_only",
            "transformer_only",
            "hybrid_no_position",
        ],
        help="Remaining ablation models only. Full hybrid with positional embedding is excluded because it already exists.",
    )

    args = parser.parse_args()

    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img_size = (args.img_size, args.img_size)

    train_gen, val_gen, test_gen, train_dir, test_dir = make_generators(
        args.data_root,
        img_size,
        args.batch_size,
        args.seed,
    )

    class_names = list(train_gen.class_indices.keys())
    num_classes = len(class_names)

    class_weights = compute_weights(train_gen)

    config = vars(args).copy()
    config.update({
        "train_dir": str(train_dir),
        "test_dir": str(test_dir),
        "class_indices": train_gen.class_indices,
        "class_weights": class_weights,
        "fairness_verification": {
            "hybrid_with_position_excluded_from_training": True,
            "validation_split": 0.20,
            "normalization": "Rescaling(1/255) inside each model",
            "label_mode": "categorical",
            "label_smoothing": args.label_smoothing,
            "class_weights_used": True,
            "early_stopping_monitor": "val_loss",
            "early_stopping_patience": args.patience,
            "reduce_lr_monitor": "val_loss",
            "reduce_lr_patience": args.lr_patience,
            "min_lr": args.min_lr,
            "augmentation_matches_full_pipeline": True,
            "checkpoint_monitor": "val_accuracy",
        }
    })

    with open(output_dir / "ablation_config_and_fairness_check.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    all_results = []

    for model_name in args.models:
        tf.keras.backend.clear_session()
        set_seed(args.seed)

        if model_name == "cnn_only":
            model = build_cnn_only(
                img_size=img_size,
                num_classes=num_classes,
                l2_reg=args.l2_reg,
            )

        elif model_name == "transformer_only":
            model = build_transformer_only(
                img_size=img_size,
                num_classes=num_classes,
                patch_size=args.patch_size,
                embed_dim=args.embed_dim,
                num_heads=args.num_heads,
                depth=args.depth,
                ff_dim=args.ff_dim,
                dropout=args.dropout,
                l2_reg=args.l2_reg,
            )

        elif model_name == "hybrid_no_position":
            model = build_hybrid_no_position(
                img_size=img_size,
                num_classes=num_classes,
                embed_dim=args.embed_dim,
                num_heads=args.num_heads,
                depth=args.depth,
                ff_dim=args.ff_dim,
                dropout=args.dropout,
                l2_reg=args.l2_reg,
            )

        else:
            raise ValueError(model_name)

        val_metrics, test_metrics = run_single_model(
            model_name=model_name,
            model=model,
            train_gen=train_gen,
            val_gen=val_gen,
            test_gen=test_gen,
            class_names=class_names,
            class_weights=class_weights,
            args=args,
        )

        all_results.append(val_metrics)
        all_results.append(test_metrics)

        partial_df = pd.DataFrame(all_results)
        partial_df.to_csv(output_dir / "ablation_results_partial.csv", index=False)
        plot_ablation_chart(partial_df, output_dir, filename="ablation_comparison_partial.png")

    results_df = pd.DataFrame(all_results)

    results_df.to_csv(output_dir / "ablation_results_remaining_three.csv", index=False)

    # Output chart requested by user.
    plot_ablation_chart(results_df, output_dir, filename="ablation_comparison_remaining_three.png")

    print("\nSaved ablation results to:", output_dir)
    print(results_df.to_string(index=False))

    print("\nFairness check:")
    print("- Hybrid with positional embedding is excluded from training.")
    print("- Existing full hybrid result should be added manually to the final paper table.")
    print("- Data split, augmentation, normalization, optimizer, label smoothing, class weights, and callbacks match the full pipeline settings.")

    print("\nExisting full hybrid-with-position reference:")
    print("accuracy=0.981119, macro_f1=0.981035, macro_auc_ovr=0.997548")


if __name__ == "__main__":
    main()
