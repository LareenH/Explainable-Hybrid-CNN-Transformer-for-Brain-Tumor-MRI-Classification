
"""
Grad-CAM Explainability for Hybrid CNN-Transformer Brain Tumor Classifier
========================================================================

Colab/local version matching the exact TransformerBlock used in:
experiment_cnn_transformer_full_pipeline.py

Features:
- Supports --gradcam_layer so you can compare token_projection vs conv2d_7.
- Saves PNG Grad-CAM panels.
- Saves gradcam_examples.csv.
- Works with the Colab-trained model saved in Google Drive.
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers, regularizers


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


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


def find_test_dir(data_root):
    data_root = Path(data_root)
    for name in ["Testing", "testing", "Test", "test"]:
        candidate = data_root / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find test folder inside {data_root}. Expected Testing/testing/Test/test."
    )


def load_class_names(model_path, test_dir):
    model_dir = Path(model_path).parent

    for json_name in ["class_indices.json", "experiment_config.json"]:
        path = model_dir / json_name
        if not path.exists():
            continue
        try:
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

        except Exception:
            pass

    return sorted([p.name for p in Path(test_dir).iterdir() if p.is_dir()])


def collect_images(test_dir, samples_per_class=3, seed=42):
    random.seed(seed)
    selected = []

    for class_dir in sorted([p for p in Path(test_dir).iterdir() if p.is_dir()]):
        files = [
            p for p in class_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        random.shuffle(files)
        selected.extend(files[:samples_per_class])

    return selected


def list_conv_layers(model):
    rows = []
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            rows.append((layer.name, str(layer.output.shape)))
    return rows


def find_gradcam_layer(model, requested_layer=None):
    if requested_layer:
        try:
            model.get_layer(requested_layer)
            return requested_layer
        except Exception as exc:
            available = list_conv_layers(model)
            raise ValueError(
                f"Requested layer '{requested_layer}' was not found.\n"
                f"Available Conv2D layers: {available}"
            ) from exc

    for name in ["conv2d_7", "token_projection"]:
        try:
            model.get_layer(name)
            return name
        except Exception:
            pass

    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name

    raise ValueError("No suitable Conv2D layer found for Grad-CAM.")


def preprocess_image(image_path, img_size):
    img = tf.keras.utils.load_img(image_path, target_size=img_size)
    arr = tf.keras.utils.img_to_array(img)
    arr_batch = np.expand_dims(arr, axis=0)
    return arr_batch, arr.astype("uint8")


def make_gradcam_heatmap(img_array, model, gradcam_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(gradcam_layer_name).output,
            model.output,
        ],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array, training=False)

        if pred_index is None:
            pred_index = tf.argmax(predictions[0])

        class_score = predictions[:, pred_index]

    grads = tape.gradient(class_score, conv_outputs)

    if grads is None:
        raise RuntimeError("Could not compute gradients. Try another convolutional layer.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy(), predictions.numpy()[0]


def overlay_heatmap(original_img, heatmap, alpha=0.45):
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap)

    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(original_img, 1 - alpha, heatmap_color, alpha, 0)

    return heatmap_color, overlay


def save_gradcam_panel(
    original_img,
    heatmap_img,
    overlay_img,
    true_class,
    pred_class,
    confidence,
    image_path,
    output_dir,
    layer_name,
):
    image_path = Path(image_path)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(original_img)
    plt.title(f"Original\nTrue: {true_class}")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(heatmap_img)
    plt.title(f"Grad-CAM\nLayer: {layer_name}")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(overlay_img)
    plt.title(f"Overlay\nPred: {pred_class} ({confidence:.2%})")
    plt.axis("off")

    plt.tight_layout()

    safe_name = f"{true_class}_{image_path.stem}_{layer_name}_gradcam.png"
    safe_name = safe_name.replace(" ", "_").replace("/", "_")

    out_path = Path(output_dir) / safe_name
    plt.savefig(out_path, dpi=300)
    plt.close()

    return out_path


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--samples_per_class", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument(
        "--gradcam_layer",
        default=None,
        help="Layer for Grad-CAM, e.g. conv2d_7 or token_projection. If omitted, tries conv2d_7 first.",
    )
    parser.add_argument(
        "--list_layers_only",
        action="store_true",
        help="Only load model and print available Conv2D layers.",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model:", args.model_path)

    model = tf.keras.models.load_model(
        args.model_path,
        compile=False,
        custom_objects={"TransformerBlock": TransformerBlock},
        safe_mode=False,
    )

    print("\nAvailable Conv2D layers:")
    for name, shape in list_conv_layers(model):
        print(f"  {name}: {shape}")

    if args.list_layers_only:
        return

    test_dir = find_test_dir(args.data_root)
    class_names = load_class_names(args.model_path, test_dir)
    gradcam_layer_name = find_gradcam_layer(model, args.gradcam_layer)

    print("\nUsing test directory:", test_dir)
    print("Class names:", class_names)
    print("Grad-CAM layer:", gradcam_layer_name)

    image_paths = collect_images(
        test_dir,
        samples_per_class=args.samples_per_class,
        seed=args.seed,
    )

    rows = []

    for image_path in image_paths:
        image_path = Path(image_path)
        true_class = image_path.parent.name

        img_array, original_img = preprocess_image(
            image_path,
            (args.img_size, args.img_size),
        )

        heatmap, probs = make_gradcam_heatmap(
            img_array,
            model,
            gradcam_layer_name,
        )

        pred_idx = int(np.argmax(probs))
        pred_class = class_names[pred_idx]
        confidence = float(probs[pred_idx])

        heatmap_img, overlay_img = overlay_heatmap(
            original_img,
            heatmap,
            alpha=args.alpha,
        )

        saved_path = save_gradcam_panel(
            original_img=original_img,
            heatmap_img=heatmap_img,
            overlay_img=overlay_img,
            true_class=true_class,
            pred_class=pred_class,
            confidence=confidence,
            image_path=image_path,
            output_dir=output_dir,
            layer_name=gradcam_layer_name,
        )

        rows.append({
            "image": str(image_path),
            "true_class": true_class,
            "predicted_class": pred_class,
            "confidence": confidence,
            "correct": true_class == pred_class,
            "gradcam_layer": gradcam_layer_name,
            "gradcam_file": str(saved_path),
        })

        print(f"Saved: {saved_path}")

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "gradcam_examples.csv", index=False)

    print("\nSaved Grad-CAM results to:", output_dir)
    print("CSV:", output_dir / "gradcam_examples.csv")


if __name__ == "__main__":
    main()
