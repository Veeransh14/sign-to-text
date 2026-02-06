"""
Real-time multi-model inference and comparison for Sign Language Recognition.

Loads all 4 trained models, runs them on live webcam input,
and displays predictions, confidence scores, specifications,
and limitations side-by-side.

Usage:
    python model_comparison.py

Prerequisites:
    Run train_models.py first to generate trained models in models/
"""

import os
import json
import time
import numpy as np
import cv2
from collections import deque

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

try:
    from cvzone.HandTrackingModule import HandDetector
except ImportError:
    print("Error: cvzone is required. Install with: pip install cvzone")
    exit(1)

# ─── Configuration ────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
IMG_SIZE = 64
SEQ_LEN = 10
NUM_CLASSES = 26
LETTERS = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

# Hand skeleton drawing connections (same as original project)
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17),                                # Palm base
]

# Model limitations
LIMITATIONS = {
    'cnn2d_model': [
        "No temporal context",
        "Single-frame only",
        "Sensitive to orientation",
    ],
    'cnn3d_model': [
        "Requires frame buffer",
        "Higher latency",
        "Needs more data",
    ],
    'lstm_model': [
        "Sequential processing",
        "Long-range dependency issues",
        "Slower training",
    ],
    'transformer_model': [
        "Highest memory usage",
        "Overkill for short sequences",
        "Quadratic complexity",
    ],
}

MODEL_TYPES = {
    'cnn2d_model': '2D CNN',
    'cnn3d_model': '3D CNN',
    'lstm_model': 'CNN-LSTM',
    'transformer_model': 'CNN-Transformer',
}


# ─── Custom layers (needed for loading) ──────────────────────────────────────

class PositionalEncoding(layers.Layer):
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


# ─── Model Loading ────────────────────────────────────────────────────────────

def load_all_models():
    """Load all trained models. Returns dict of {name: model} for available models."""
    loaded = {}
    custom_objects = {'PositionalEncoding': PositionalEncoding}
    model_files = {
        'cnn2d_model': 'cnn2d_model.h5',
        'cnn3d_model': 'cnn3d_model.h5',
        'lstm_model': 'lstm_model.h5',
        'transformer_model': 'transformer_model.h5',
    }
    for name, fname in model_files.items():
        path = os.path.join(MODEL_DIR, fname)
        if os.path.exists(path):
            try:
                model = keras.models.load_model(path, custom_objects=custom_objects)
                loaded[name] = model
                print(f"  Loaded {name} from {fname}")
            except Exception as e:
                print(f"  Warning: Failed to load {name}: {e}")
        else:
            print(f"  Skipping {name} — file not found: {fname}")
    return loaded


def load_comparison_results():
    """Load comparison metrics from JSON."""
    path = os.path.join(MODEL_DIR, 'comparison_results.json')
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None


# ─── Hand Skeleton ────────────────────────────────────────────────────────────

def draw_skeleton(landmarks, canvas_size=400):
    """Draw hand skeleton on white background from landmarks list."""
    canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255

    if landmarks is None or len(landmarks) < 21:
        return canvas

    pts = landmarks

    # Draw connections
    for i, j in CONNECTIONS:
        if i < len(pts) and j < len(pts):
            cv2.line(canvas, (pts[i][0], pts[i][1]), (pts[j][0], pts[j][1]),
                     (0, 180, 0), 3)

    # Draw landmarks
    for p in pts:
        cv2.circle(canvas, (p[0], p[1]), 4, (0, 0, 200), -1)

    return canvas


def extract_skeleton_image(frame, hd, hd2, offset=58):
    """Extract hand skeleton from frame. Returns (skeleton_img_64, skeleton_img_400, success)."""
    hands = hd.findHands(frame, draw=False, flipType=True)

    if not hands or len(hands[0]) == 0:
        return None, None, False

    hand = hands[0]
    if isinstance(hand, list):
        if len(hand) == 0:
            return None, None, False
        hand = hand[0]

    if 'bbox' not in hand:
        return None, None, False

    x, y, w, h = hand['bbox']
    x1 = max(0, x - offset)
    y1 = max(0, y - offset)
    x2 = min(frame.shape[1], x + w + offset)
    y2 = min(frame.shape[0], y + h + offset)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None, None, False

    # Detect hand in cropped region for better landmark accuracy
    hands2 = hd2.findHands(crop, draw=False, flipType=True)
    if not hands2 or len(hands2[0]) == 0:
        return None, None, False

    hand2 = hands2[0]
    if isinstance(hand2, list):
        if len(hand2) == 0:
            return None, None, False
        hand2 = hand2[0]

    if 'lmList' not in hand2:
        return None, None, False

    lm = hand2['lmList']

    # Scale landmarks to 400x400 canvas
    crop_h, crop_w = crop.shape[:2]
    scaled_landmarks = []
    for pt in lm:
        sx = int(pt[0] * 400 / crop_w)
        sy = int(pt[1] * 400 / crop_h)
        scaled_landmarks.append((sx, sy))

    skeleton_400 = draw_skeleton(scaled_landmarks, 400)
    skeleton_64 = cv2.resize(skeleton_400, (IMG_SIZE, IMG_SIZE))

    return skeleton_64, skeleton_400, True


# ─── Display Helpers ──────────────────────────────────────────────────────────

def put_text(img, text, pos, scale=0.5, color=(0, 0, 0), thickness=1):
    """Put anti-aliased text on image."""
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_info_panel(width, height, predictions, comparison_results, best_model_name, sentence):
    """Draw the right-side info panel with predictions, specs, and limitations."""
    panel = np.ones((height, width, 3), dtype=np.uint8) * 245  # Light gray background

    y = 30
    put_text(panel, "MODEL PREDICTIONS", (15, y), 0.7, (40, 40, 40), 2)
    y += 10
    cv2.line(panel, (15, y), (width - 15, y), (180, 180, 180), 1)
    y += 30

    for model_name, (letter, confidence, inf_time) in predictions.items():
        display_name = MODEL_TYPES.get(model_name, model_name)
        is_best = (model_name == best_model_name)
        color = (0, 130, 0) if is_best else (60, 60, 60)

        label = f"{display_name}: {letter} ({confidence:.1f}%)"
        if is_best:
            label += "  << BEST"
        put_text(panel, label, (20, y), 0.55, color, 2 if is_best else 1)
        y += 18
        put_text(panel, f"  Inference: {inf_time:.1f}ms", (20, y), 0.4, (120, 120, 120), 1)
        y += 28

    # Specifications table
    y += 10
    cv2.line(panel, (15, y), (width - 15, y), (180, 180, 180), 1)
    y += 25
    put_text(panel, "SPECIFICATIONS", (15, y), 0.6, (40, 40, 40), 2)
    y += 25

    if comparison_results:
        # Table header
        put_text(panel, f"{'Model':<16} {'Acc%':>6} {'F1%':>6} {'Params':>8}", (20, y), 0.4, (80, 80, 80), 1)
        y += 5
        cv2.line(panel, (20, y), (width - 20, y), (200, 200, 200), 1)
        y += 18

        for m in comparison_results.get('models', []):
            name = MODEL_TYPES.get(m['model_name'], m['model_name'])
            params = m['param_count']
            params_str = f"{params / 1000:.0f}K" if params < 1_000_000 else f"{params / 1_000_000:.1f}M"
            is_best = (m['model_name'] == best_model_name)
            color = (0, 130, 0) if is_best else (60, 60, 60)
            put_text(panel, f"{name:<16} {m['accuracy']:>5.1f} {m['f1_score']:>5.1f} {params_str:>8}",
                     (20, y), 0.4, color, 1)
            y += 20
    else:
        put_text(panel, "(Run train_models.py first)", (20, y), 0.4, (150, 100, 100), 1)
        y += 20

    # Limitations
    y += 15
    cv2.line(panel, (15, y), (width - 15, y), (180, 180, 180), 1)
    y += 25
    put_text(panel, "LIMITATIONS", (15, y), 0.6, (40, 40, 40), 2)
    y += 25

    for model_name in predictions:
        display_name = MODEL_TYPES.get(model_name, model_name)
        put_text(panel, f"{display_name}:", (20, y), 0.42, (60, 60, 60), 1)
        y += 18
        for lim in LIMITATIONS.get(model_name, []):
            put_text(panel, f"  - {lim}", (25, y), 0.35, (120, 80, 80), 1)
            y += 16
        y += 8

    # Sentence at bottom
    y = height - 40
    cv2.line(panel, (15, y - 10), (width - 15, y - 10), (180, 180, 180), 1)
    put_text(panel, f"Sentence: {sentence}", (15, y + 5), 0.5, (30, 30, 30), 1)
    best_display = MODEL_TYPES.get(best_model_name, best_model_name)
    put_text(panel, f"[Best: {best_display}]", (15, y + 25), 0.4, (0, 130, 0), 1)

    return panel


# ─── Main Loop ────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  SIGN LANGUAGE — MULTI-MODEL COMPARISON (Live)")
    print("=" * 60)

    # Load models
    print("\nLoading models...")
    all_models = load_all_models()
    if not all_models:
        print("No trained models found! Run train_models.py first.")
        return

    comparison_results = load_comparison_results()

    # Determine best model
    best_model_name = None
    if comparison_results:
        best_model_name = comparison_results.get('best_model')
    if best_model_name not in all_models:
        best_model_name = list(all_models.keys())[0]

    # Separate single-frame and sequence models
    single_frame_models = {}
    sequence_models = {}
    for name, model in all_models.items():
        input_shape = model.input_shape
        if len(input_shape) == 4:  # (batch, h, w, c)
            single_frame_models[name] = model
        else:  # (batch, seq, h, w, c)
            sequence_models[name] = model

    # Camera setup
    print("\nOpening camera...")
    cap = None
    for idx in range(10):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            print(f"  Camera opened at index {idx}")
            break
        cap.release()
    else:
        print("Error: No camera found!")
        return

    # Hand detectors
    hd = HandDetector(detectionCon=0.8, maxHands=1)
    hd2 = HandDetector(detectionCon=0.8, maxHands=1)

    # Frame buffer for sequence models
    frame_buffer = deque(maxlen=SEQ_LEN)

    # State
    sentence = ""
    predictions = {}
    prev_letter = ""
    stable_count = 0
    STABLE_THRESHOLD = 8  # Need N consistent frames to register a letter

    print("\nStarting live inference. Press 'q' to quit, 'c' to clear sentence, 's' to speak.")
    print("-" * 60)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        display_frame = frame.copy()

        # Extract skeleton
        skeleton_64, skeleton_400, success = extract_skeleton_image(frame, hd, hd2)

        if success and skeleton_64 is not None:
            # Normalize for model input
            input_img = skeleton_64.astype(np.float32) / 255.0
            frame_buffer.append(input_img)

            # Run single-frame models
            for name, model in single_frame_models.items():
                batch = np.expand_dims(input_img, axis=0)
                start_t = time.time()
                pred = model.predict(batch, verbose=0)[0]
                inf_time = (time.time() - start_t) * 1000
                idx = np.argmax(pred)
                confidence = pred[idx] * 100
                predictions[name] = (LETTERS[idx], confidence, inf_time)

            # Run sequence models (only when buffer is full)
            if len(frame_buffer) == SEQ_LEN:
                seq_batch = np.expand_dims(np.array(frame_buffer), axis=0)
                for name, model in sequence_models.items():
                    start_t = time.time()
                    pred = model.predict(seq_batch, verbose=0)[0]
                    inf_time = (time.time() - start_t) * 1000
                    idx = np.argmax(pred)
                    confidence = pred[idx] * 100
                    predictions[name] = (LETTERS[idx], confidence, inf_time)
            else:
                for name in sequence_models:
                    predictions[name] = ('...', 0.0, 0.0)

            # Use best model's prediction for sentence building
            if best_model_name in predictions:
                current_letter = predictions[best_model_name][0]
                if current_letter == prev_letter and current_letter != '...':
                    stable_count += 1
                else:
                    stable_count = 0
                prev_letter = current_letter

                if stable_count == STABLE_THRESHOLD:
                    sentence += current_letter
        else:
            skeleton_400 = np.ones((400, 400, 3), dtype=np.uint8) * 255
            for name in all_models:
                if name not in predictions:
                    predictions[name] = ('-', 0.0, 0.0)

        # Build display
        # Left: camera feed resized
        cam_display = cv2.resize(display_frame, (420, 340))

        # Middle: skeleton
        skel_display = cv2.resize(skeleton_400, (340, 340))

        # Right: info panel
        info_panel = draw_info_panel(440, 700, predictions, comparison_results,
                                     best_model_name, sentence)

        # Compose final display
        left_col = np.ones((700, 420, 3), dtype=np.uint8) * 30
        left_col[10:350, :] = cam_display
        # Add label under camera
        put_text(left_col, "Live Camera Feed", (130, 370), 0.5, (200, 200, 200), 1)

        mid_col = np.ones((700, 340, 3), dtype=np.uint8) * 30
        mid_col[10:350, :] = skel_display
        put_text(mid_col, "Hand Skeleton", (100, 370), 0.5, (200, 200, 200), 1)

        # Instructions at bottom of left/mid columns
        put_text(left_col, "Controls:", (10, 420), 0.45, (180, 180, 180), 1)
        put_text(left_col, "  q - Quit", (10, 445), 0.4, (150, 150, 150), 1)
        put_text(left_col, "  c - Clear sentence", (10, 465), 0.4, (150, 150, 150), 1)
        put_text(left_col, "  s - Speak sentence", (10, 485), 0.4, (150, 150, 150), 1)

        # Current best prediction large display
        if best_model_name in predictions:
            letter, conf, _ = predictions[best_model_name]
            put_text(mid_col, f"Prediction: {letter}", (50, 430), 0.8, (0, 255, 100), 2)
            put_text(mid_col, f"Confidence: {conf:.1f}%", (70, 465), 0.55, (200, 200, 200), 1)

        # Buffer status
        buf_pct = len(frame_buffer) / SEQ_LEN * 100
        put_text(mid_col, f"Buffer: {len(frame_buffer)}/{SEQ_LEN} ({buf_pct:.0f}%)",
                 (80, 510), 0.45, (150, 150, 150), 1)

        # Combine
        combined = np.hstack([left_col, mid_col, info_panel])

        cv2.imshow("Sign Language - Multi-Model Comparison", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            sentence = ""
        elif key == ord('s'):
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.setProperty('rate', 120)
                engine.say(sentence)
                engine.runAndWait()
            except Exception as e:
                print(f"Speech error: {e}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nFinal sentence: {sentence}")
    print("Done.")


if __name__ == '__main__':
    main()
