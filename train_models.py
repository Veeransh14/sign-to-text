"""
Train and compare 4 models for Sign Language Recognition:
  1. Enhanced 2D CNN
  2. 3D CNN (temporal sequences)
  3. CNN-LSTM (temporal sequences)
  4. CNN-Transformer (temporal sequences)

All models predict 26 classes (A-Z) directly.
Results are saved to models/ directory with a comparison JSON.

Usage:
    python train_models.py
"""

import os
import json
import time
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2

# ─── Configuration ────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), "AtoZ_3.1")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
IMG_SIZE = 64
SEQ_LEN = 10
SEQ_STRIDE = 5
NUM_CLASSES = 26
EPOCHS = 50
BATCH_SIZE_SINGLE = 32
BATCH_SIZE_SEQ = 16
LETTERS = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

os.makedirs(MODEL_DIR, exist_ok=True)


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_single_frame_data():
    """Load individual images for 2D CNN. Returns images and labels."""
    images, labels = [], []
    for idx, letter in enumerate(LETTERS):
        letter_dir = os.path.join(DATA_DIR, letter)
        if not os.path.isdir(letter_dir):
            print(f"  Warning: directory not found for letter {letter}")
            continue
        files = sorted(
            [f for f in os.listdir(letter_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))],
            key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else 0
        )
        for fname in files:
            img = cv2.imread(os.path.join(letter_dir, fname))
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img.astype(np.float32) / 255.0
            images.append(img)
            labels.append(idx)
    return np.array(images), np.array(labels)


def load_sequence_data():
    """Create sequences of consecutive frames for temporal models."""
    sequences, labels = [], []
    for idx, letter in enumerate(LETTERS):
        letter_dir = os.path.join(DATA_DIR, letter)
        if not os.path.isdir(letter_dir):
            continue
        files = sorted(
            [f for f in os.listdir(letter_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))],
            key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else 0
        )
        # Load all images for this letter
        letter_images = []
        for fname in files:
            img = cv2.imread(os.path.join(letter_dir, fname))
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img.astype(np.float32) / 255.0
            letter_images.append(img)

        # Create sequences with sliding window
        for start in range(0, len(letter_images) - SEQ_LEN + 1, SEQ_STRIDE):
            seq = letter_images[start:start + SEQ_LEN]
            sequences.append(np.array(seq))
            labels.append(idx)

    return np.array(sequences), np.array(labels)


def augment_image(image):
    """Simple data augmentation for a single image."""
    # Random brightness
    if np.random.random() > 0.5:
        factor = np.random.uniform(0.8, 1.2)
        image = np.clip(image * factor, 0, 1)
    # Random rotation (small)
    if np.random.random() > 0.5:
        h, w = image.shape[:2]
        angle = np.random.uniform(-15, 15)
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h), borderValue=(1, 1, 1))
    return image


# ─── Model Definitions ────────────────────────────────────────────────────────

def build_cnn2d():
    """Enhanced 2D CNN for single-frame classification."""
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),

        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ], name='CNN_2D')
    return model


def build_cnn3d():
    """3D CNN operating on sequences of frames."""
    model = models.Sequential([
        layers.Input(shape=(SEQ_LEN, IMG_SIZE, IMG_SIZE, 3)),

        layers.Conv3D(32, (3, 3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling3D((2, 2, 2)),

        layers.Conv3D(64, (3, 3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling3D((2, 2, 2)),

        layers.Conv3D(128, (3, 3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.GlobalAveragePooling3D(),

        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ], name='CNN_3D')
    return model


def _build_cnn_feature_extractor():
    """Shared CNN feature extractor for LSTM and Transformer."""
    inp = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = layers.Conv2D(32, (3, 3), padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    return keras.Model(inp, x, name='cnn_feature_extractor')


def build_cnn_lstm():
    """CNN feature extractor + LSTM for sequence classification."""
    cnn_extractor = _build_cnn_feature_extractor()

    inp = layers.Input(shape=(SEQ_LEN, IMG_SIZE, IMG_SIZE, 3))
    # Apply CNN to each frame
    x = layers.TimeDistributed(cnn_extractor)(inp)
    # LSTM layers
    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.LSTM(64)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    model = keras.Model(inp, x, name='CNN_LSTM')
    return model


class PositionalEncoding(layers.Layer):
    """Learned positional encoding for transformer input."""
    def __init__(self, seq_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.d_model = d_model
        self.pos_embedding = layers.Embedding(input_dim=seq_len, output_dim=d_model)

    def call(self, x):
        positions = tf.range(start=0, limit=self.seq_len, delta=1)
        pos_enc = self.pos_embedding(positions)
        return x + pos_enc

    def get_config(self):
        config = super().get_config()
        config.update({'seq_len': self.seq_len, 'd_model': self.d_model})
        return config


def build_cnn_transformer():
    """CNN feature extractor + Transformer encoder for sequence classification."""
    cnn_extractor = _build_cnn_feature_extractor()

    inp = layers.Input(shape=(SEQ_LEN, IMG_SIZE, IMG_SIZE, 3))
    # Apply CNN to each frame
    x = layers.TimeDistributed(cnn_extractor)(inp)

    # Positional encoding
    x = PositionalEncoding(SEQ_LEN, 128)(x)

    # Transformer encoder block
    attn_output = layers.MultiHeadAttention(
        num_heads=4, key_dim=32
    )(x, x)
    x = layers.Add()([x, attn_output])
    x = layers.LayerNormalization()(x)

    ffn_output = layers.Dense(256, activation='relu')(x)
    ffn_output = layers.Dense(128)(ffn_output)
    x = layers.Add()([x, ffn_output])
    x = layers.LayerNormalization()(x)

    # Classification head
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    model = keras.Model(inp, x, name='CNN_Transformer')
    return model


# ─── Training ─────────────────────────────────────────────────────────────────

def train_model(model, X_train, y_train, X_val, y_val, model_name, batch_size):
    """Train a model with standard callbacks. Returns training history."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model_path = os.path.join(MODEL_DIR, f"{model_name}.h5")

    cb = [
        callbacks.ModelCheckpoint(
            model_path, monitor='val_accuracy',
            save_best_only=True, verbose=1
        ),
        callbacks.EarlyStopping(
            monitor='val_accuracy', patience=10,
            restore_best_weights=True, verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=5, min_lr=1e-6, verbose=1
        ),
    ]

    y_train_cat = to_categorical(y_train, NUM_CLASSES)
    y_val_cat = to_categorical(y_val, NUM_CLASSES)

    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=EPOCHS,
        batch_size=batch_size,
        callbacks=cb,
        verbose=1
    )
    return history


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model and return detailed metrics."""
    y_test_cat = to_categorical(y_test, NUM_CLASSES)

    # Accuracy from Keras
    loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)

    # Predictions + timing
    start = time.time()
    y_pred_probs = model.predict(X_test, verbose=0)
    total_time = time.time() - start
    avg_inference_ms = (total_time / len(X_test)) * 1000

    y_pred = np.argmax(y_pred_probs, axis=1)

    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    param_count = model.count_params()
    model_path = os.path.join(MODEL_DIR, f"{model_name}.h5")
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024) if os.path.exists(model_path) else 0

    report = classification_report(y_test, y_pred, target_names=LETTERS, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        'model_name': model_name,
        'accuracy': round(acc * 100, 2),
        'precision': round(precision * 100, 2),
        'recall': round(recall * 100, 2),
        'f1_score': round(f1 * 100, 2),
        'loss': round(loss, 4),
        'avg_inference_ms': round(avg_inference_ms, 2),
        'param_count': param_count,
        'model_size_mb': round(model_size_mb, 2),
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
    }
    return metrics


# ─── Main ─────────────────────────────────────────────────────────────────────

def print_comparison_table(all_metrics):
    """Print a formatted comparison table."""
    print("\n" + "=" * 90)
    print("MODEL COMPARISON RESULTS")
    print("=" * 90)
    header = f"{'Model':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Params':>10} {'Inf(ms)':>10}"
    print(header)
    print("-" * 90)

    best_idx = max(range(len(all_metrics)), key=lambda i: all_metrics[i]['f1_score'])

    for i, m in enumerate(all_metrics):
        marker = " << BEST" if i == best_idx else ""
        params_str = f"{m['param_count'] / 1000:.0f}K" if m['param_count'] < 1_000_000 else f"{m['param_count'] / 1_000_000:.1f}M"
        print(
            f"{m['model_name']:<20} {m['accuracy']:>9.2f}% {m['precision']:>9.2f}% "
            f"{m['recall']:>9.2f}% {m['f1_score']:>9.2f}% {params_str:>10} "
            f"{m['avg_inference_ms']:>8.2f}ms{marker}"
        )

    print("=" * 90)
    print(f"\nBest Model: {all_metrics[best_idx]['model_name']} (F1: {all_metrics[best_idx]['f1_score']:.2f}%)")
    return all_metrics[best_idx]['model_name']


def print_limitations():
    """Print limitations and specifications for each model."""
    print("\n" + "=" * 90)
    print("MODEL SPECIFICATIONS & LIMITATIONS")
    print("=" * 90)

    specs = {
        'cnn2d_model': {
            'type': '2D Convolutional Neural Network',
            'input': f'Single frame ({IMG_SIZE}x{IMG_SIZE}x3)',
            'strengths': [
                'Fast inference (single frame)',
                'Simple architecture, easy to train',
                'Works well with limited data',
                'Low memory footprint',
            ],
            'limitations': [
                'No temporal context — each frame classified independently',
                'Sensitive to hand orientation and position',
                'Cannot capture motion-based signs (J, Z)',
                'May confuse visually similar static poses',
            ],
        },
        'cnn3d_model': {
            'type': '3D Convolutional Neural Network',
            'input': f'Sequence of {SEQ_LEN} frames ({SEQ_LEN}x{IMG_SIZE}x{IMG_SIZE}x3)',
            'strengths': [
                'Captures spatio-temporal features',
                'Can distinguish motion-based signs',
                'Learns temporal patterns in gesture transitions',
            ],
            'limitations': [
                'Requires buffering multiple frames — adds latency',
                'Higher memory and compute requirements',
                'Needs more training data for temporal patterns',
                'Slower inference than 2D CNN',
                'Current dataset has static poses — temporal benefit limited',
            ],
        },
        'lstm_model': {
            'type': 'CNN + LSTM (Long Short-Term Memory)',
            'input': f'Sequence of {SEQ_LEN} frames → CNN features → LSTM',
            'strengths': [
                'Good at modeling sequential dependencies',
                'CNN extracts spatial features; LSTM models time',
                'Handles variable-length gesture dynamics',
            ],
            'limitations': [
                'Sequential processing — cannot be parallelized',
                'May struggle with long-range dependencies',
                'Training can be slow due to recurrence',
                'Vanishing gradient possible with longer sequences',
                'Requires sequence buffering at inference time',
            ],
        },
        'transformer_model': {
            'type': 'CNN + Transformer Encoder',
            'input': f'Sequence of {SEQ_LEN} frames → CNN features → Self-Attention',
            'strengths': [
                'Parallel processing of all frames via self-attention',
                'Captures global temporal relationships',
                'State-of-the-art architecture for sequence modeling',
                'Positional encoding preserves temporal order',
            ],
            'limitations': [
                'Highest memory usage due to attention matrices',
                'Overkill for short sequences — benefit diminishes',
                'Needs more training data than simpler models',
                'Slower than 2D CNN for single-frame tasks',
                'Quadratic complexity with sequence length',
            ],
        },
    }

    for name, s in specs.items():
        print(f"\n--- {name} ({s['type']}) ---")
        print(f"  Input: {s['input']}")
        print(f"  Strengths:")
        for st in s['strengths']:
            print(f"    + {st}")
        print(f"  Limitations:")
        for li in s['limitations']:
            print(f"    - {li}")

    print("\n" + "=" * 90)


def main():
    print("=" * 60)
    print("  SIGN LANGUAGE RECOGNITION — MULTI-MODEL TRAINING")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────────
    print("\n[1/6] Loading single-frame data...")
    X_single, y_single = load_single_frame_data()
    print(f"  Loaded {len(X_single)} images, shape: {X_single.shape}")

    print("\n[2/6] Loading sequence data...")
    X_seq, y_seq = load_sequence_data()
    print(f"  Loaded {len(X_seq)} sequences, shape: {X_seq.shape}")

    # ── Train/val/test split ───────────────────────────────────
    # Single-frame split
    X_s_train, X_s_temp, y_s_train, y_s_temp = train_test_split(
        X_single, y_single, test_size=0.2, random_state=42, stratify=y_single
    )
    X_s_val, X_s_test, y_s_val, y_s_test = train_test_split(
        X_s_temp, y_s_temp, test_size=0.5, random_state=42, stratify=y_s_temp
    )
    print(f"  Single-frame — Train: {len(X_s_train)}, Val: {len(X_s_val)}, Test: {len(X_s_test)}")

    # Sequence split
    X_q_train, X_q_temp, y_q_train, y_q_temp = train_test_split(
        X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq
    )
    X_q_val, X_q_test, y_q_val, y_q_test = train_test_split(
        X_q_temp, y_q_temp, test_size=0.5, random_state=42, stratify=y_q_temp
    )
    print(f"  Sequence     — Train: {len(X_q_train)}, Val: {len(X_q_val)}, Test: {len(X_q_test)}")

    # Apply simple augmentation to single-frame training data
    print("\n  Augmenting single-frame training data...")
    augmented = []
    for img in X_s_train:
        augmented.append(augment_image(img.copy()))
    X_s_train = np.concatenate([X_s_train, np.array(augmented)], axis=0)
    y_s_train = np.concatenate([y_s_train, y_s_train], axis=0)
    print(f"  Augmented single-frame train size: {len(X_s_train)}")

    all_metrics = []

    # ── Model 1: 2D CNN ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("[3/6] Training 2D CNN...")
    print("=" * 60)
    cnn2d = build_cnn2d()
    cnn2d.summary()
    train_model(cnn2d, X_s_train, y_s_train, X_s_val, y_s_val, 'cnn2d_model', BATCH_SIZE_SINGLE)
    # Reload best weights
    cnn2d = keras.models.load_model(os.path.join(MODEL_DIR, 'cnn2d_model.h5'))
    m1 = evaluate_model(cnn2d, X_s_test, y_s_test, 'cnn2d_model')
    all_metrics.append(m1)
    print(f"  2D CNN — Accuracy: {m1['accuracy']}%, F1: {m1['f1_score']}%")

    # ── Model 2: 3D CNN ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("[4/6] Training 3D CNN...")
    print("=" * 60)
    cnn3d = build_cnn3d()
    cnn3d.summary()
    train_model(cnn3d, X_q_train, y_q_train, X_q_val, y_q_val, 'cnn3d_model', BATCH_SIZE_SEQ)
    cnn3d = keras.models.load_model(os.path.join(MODEL_DIR, 'cnn3d_model.h5'))
    m2 = evaluate_model(cnn3d, X_q_test, y_q_test, 'cnn3d_model')
    all_metrics.append(m2)
    print(f"  3D CNN — Accuracy: {m2['accuracy']}%, F1: {m2['f1_score']}%")

    # ── Model 3: CNN-LSTM ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("[5/6] Training CNN-LSTM...")
    print("=" * 60)
    lstm_model = build_cnn_lstm()
    lstm_model.summary()
    train_model(lstm_model, X_q_train, y_q_train, X_q_val, y_q_val, 'lstm_model', BATCH_SIZE_SEQ)
    lstm_model = keras.models.load_model(os.path.join(MODEL_DIR, 'lstm_model.h5'))
    m3 = evaluate_model(lstm_model, X_q_test, y_q_test, 'lstm_model')
    all_metrics.append(m3)
    print(f"  CNN-LSTM — Accuracy: {m3['accuracy']}%, F1: {m3['f1_score']}%")

    # ── Model 4: CNN-Transformer ──────────────────────────────
    print("\n" + "=" * 60)
    print("[6/6] Training CNN-Transformer...")
    print("=" * 60)
    transformer_model = build_cnn_transformer()
    transformer_model.summary()
    train_model(
        transformer_model, X_q_train, y_q_train,
        X_q_val, y_q_val, 'transformer_model', BATCH_SIZE_SEQ
    )
    transformer_model = keras.models.load_model(
        os.path.join(MODEL_DIR, 'transformer_model.h5'),
        custom_objects={'PositionalEncoding': PositionalEncoding}
    )
    m4 = evaluate_model(transformer_model, X_q_test, y_q_test, 'transformer_model')
    all_metrics.append(m4)
    print(f"  CNN-Transformer — Accuracy: {m4['accuracy']}%, F1: {m4['f1_score']}%")

    # ── Comparison ─────────────────────────────────────────────
    best_model_name = print_comparison_table(all_metrics)
    print_limitations()

    # Print per-model classification reports
    for m in all_metrics:
        print(f"\n{'='*60}")
        print(f"Classification Report — {m['model_name']}")
        print(f"{'='*60}")
        print(m['classification_report'])

    # Save results
    results = {
        'best_model': best_model_name,
        'models': []
    }
    for m in all_metrics:
        entry = {k: v for k, v in m.items() if k != 'classification_report' and k != 'confusion_matrix'}
        results['models'].append(entry)

    results_path = os.path.join(MODEL_DIR, 'comparison_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_path}")
    print(f"Models saved to {MODEL_DIR}/")
    print(f"\nBest model for deployment: {best_model_name}")
    print("Done.")


if __name__ == '__main__':
    main()
