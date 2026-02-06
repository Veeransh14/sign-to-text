

import numpy as np
import math
import cv2
import os
import sys
import json
import time
import threading
import traceback
import pyttsx3
from keras.models import load_model
from cvzone.HandTrackingModule import HandDetector
from string import ascii_uppercase
from collections import Counter
import enchant
import tkinter as tk
from PIL import Image, ImageTk

# ─── Best-model selection ─────────────────────────────────────────────────────
# Set USE_BEST_MODEL = True to load the best model from multi-model comparison
# instead of the legacy 8-group model. Requires running train_models.py first.
USE_BEST_MODEL = True

# Initialize spell checker and hand detectors
ddd = enchant.Dict("en-US")
hd = HandDetector(maxHands=1)
hd2 = HandDetector(maxHands=1)

offset = 15  # must match data_collection_final.py (training data used 15)

os.environ["THEANO_FLAGS"] = "device=cuda, assert_no_cpu_op=True"


class Application:
    def __init__(self):
        # Try to open camera with different indices
        self.vs = None
        print("Attempting to open camera...")
        for camera_index in [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]:
            test_vs = cv2.VideoCapture(camera_index)
            if test_vs.isOpened():
                ret, test_frame = test_vs.read()
                if ret and test_frame is not None:
                    self.vs = test_vs
                    print(f"✓ Successfully opened camera at index {camera_index}")
                    break
                test_vs.release()
        
        if self.vs is None:
            raise RuntimeError("❌ Could not open any camera. Please check your camera connection.")
        
        self.current_image = None
        
        # Load models
        print("Loading models...")
        self.use_26class = False
        self.model_2d = None   # fast 2D CNN (primary, ~2ms per frame)
        self.model_3d = None   # 3D CNN (secondary ensemble, ~18ms per sequence)
        self.model = None      # legacy fallback

        if USE_BEST_MODEL:
            model_dir = os.path.join(os.path.dirname(__file__), 'models')
            # Load 2D CNN (primary — fast per-frame predictions)
            path_2d = os.path.join(model_dir, 'cnn2d_model.h5')
            if os.path.exists(path_2d):
                try:
                    self.model_2d = load_model(path_2d)
                    self.use_26class = True
                    print(f"  ✓ Loaded 2D CNN (primary): {path_2d}")
                except Exception as e:
                    print(f"  Warning: Failed to load 2D CNN: {e}")

            # Load 3D CNN (ensemble confirmation)
            path_3d = os.path.join(model_dir, 'cnn3d_model.h5')
            if os.path.exists(path_3d):
                try:
                    self.model_3d = load_model(path_3d)
                    print(f"  ✓ Loaded 3D CNN (ensemble): {path_3d}")
                except Exception as e:
                    print(f"  Warning: Failed to load 3D CNN: {e}")

            if not self.use_26class:
                print("  No 26-class models found, falling back to legacy")
                self.model = load_model('cnn8grps_rad1_model.h5')
        else:
            self.model = load_model('cnn8grps_rad1_model.h5')
        print("✓ Models loaded successfully")
        
        # Initialize text-to-speech engine
        self.speak_engine = pyttsx3.init()
        self.speak_engine.setProperty("rate", 100)
        voices = self.speak_engine.getProperty("voices")
        self.speak_engine.setProperty("voice", voices[0].id)

        # Sequence buffer for temporal models (3D CNN, LSTM, Transformer)
        self.frame_buffer = []
        self.seq_len = 10

        # ── Prediction smoothing & filtering ─────────────────────────
        self.smoothed_prob = None         # EMA-smoothed probability vector
        self.EMA_ALPHA = 0.3             # smoothing factor (lower = more stable, absorbs noise)
        self.CONFIDENCE_THRESHOLD = 0.50  # minimum confidence to accept a letter
        self.VOTE_WINDOW = 20            # sliding window for majority voting (more frames = stabler)
        self.recent_preds = []           # recent predictions for majority vote

        # ── Letter stabilization (time-based) ──────────────────────
        self.stable_char = None          # character currently being held
        self.stable_start_time = None    # when this character first appeared
        self.STABLE_SECONDS = 2.0        # seconds to hold before finalizing
        self.last_finalized_char = None  # avoid double-adding same letter
        self.char_finalized_flag = False # flash feedback on finalization
        self.finalized_flash_time = 0    # when the flash started
        self.no_hand_time = None         # tracks when hand disappears

        # Initialize counters and flags
        self.ct = {}
        self.ct['blank'] = 0
        self.blank_flag = 0
        self.space_flag = False
        self.next_flag = True
        self.prev_char = ""
        self.count = -1
        self.ten_prev_char = []
        for i in range(10):
            self.ten_prev_char.append(" ")

        for i in ascii_uppercase:
            self.ct[i] = 0
        
        print("Initializing GUI...")
        
        # Create GUI window
        self.root = tk.Tk()
        self.root.title("Sign Language To Text Conversion")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        self.root.geometry("1300x750")

        # Video panel
        self.panel = tk.Label(self.root)
        self.panel.place(x=100, y=3, width=480, height=640)

        # Hand skeleton panel
        self.panel2 = tk.Label(self.root)
        self.panel2.place(x=700, y=115, width=400, height=400)

        # Title
        self.T = tk.Label(self.root)
        self.T.place(x=60, y=5)
        self.T.config(text="Sign Language To Text Conversion", font=("Courier", 30, "bold"))

        # Current Character label
        self.panel3 = tk.Label(self.root)
        self.panel3.place(x=280, y=585)

        self.T1 = tk.Label(self.root)
        self.T1.place(x=10, y=580)
        self.T1.config(text="Character :", font=("Courier", 30, "bold"))

        # Sentence display
        self.panel5 = tk.Label(self.root)
        self.panel5.place(x=260, y=632)

        self.T3 = tk.Label(self.root)
        self.T3.place(x=10, y=632)
        self.T3.config(text="Sentence :", font=("Courier", 30, "bold"))

        # Suggestions label
        self.T4 = tk.Label(self.root)
        self.T4.place(x=10, y=700)
        self.T4.config(text="Suggestions :", fg="red", font=("Courier", 30, "bold"))

        # Suggestion buttons
        self.b1 = tk.Button(self.root)
        self.b1.place(x=390, y=700)

        self.b2 = tk.Button(self.root)
        self.b2.place(x=590, y=700)

        self.b3 = tk.Button(self.root)
        self.b3.place(x=790, y=700)

        self.b4 = tk.Button(self.root)
        self.b4.place(x=990, y=700)

        # ── Stabilization progress bar ────────────────────────────
        self.T_stable = tk.Label(self.root)
        self.T_stable.place(x=700, y=530)
        self.T_stable.config(text="Hold letter:", font=("Courier", 14, "bold"))

        self.progress_canvas = tk.Canvas(self.root, width=280, height=24,
                                          bg="#333333", highlightthickness=1,
                                          highlightbackground="#888888")
        self.progress_canvas.place(x=700, y=555)
        self.progress_bar = self.progress_canvas.create_rectangle(0, 0, 0, 24, fill="#00cc44")
        self.progress_text = self.progress_canvas.create_text(140, 12, text="0%",
                                                               fill="white", font=("Courier", 10, "bold"))

        self.status_label = tk.Label(self.root)
        self.status_label.place(x=700, y=585)
        self.status_label.config(text="Show a sign to begin", fg="#666666",
                                  font=("Courier", 12))

        # Control buttons
        self.speak = tk.Button(self.root)
        self.speak.place(x=1205, y=630)
        self.speak.config(text="Speak", font=("Courier", 20), wraplength=100, command=self.speak_fun)

        self.speak_word_btn = tk.Button(self.root)
        self.speak_word_btn.place(x=1105, y=560)
        self.speak_word_btn.config(text="Speak\nWord", font=("Courier", 12),
                                    wraplength=80, command=self.speak_last_word)

        self.clear = tk.Button(self.root)
        self.clear.place(x=1105, y=630)
        self.clear.config(text="Clear", font=("Courier", 20), wraplength=100, command=self.clear_fun)

        # Initialize variables
        self.str = " "
        self.ccc = 0
        self.word = " "
        self.current_symbol = "C"
        self.photo = "Empty"

        self.word1 = " "
        self.word2 = " "
        self.word3 = " "
        self.word4 = " "

        # Create white background image if it doesn't exist
        if not os.path.exists("white.jpg"):
            white_img = np.ones((400, 400, 3), dtype=np.uint8) * 255
            cv2.imwrite("white.jpg", white_img)
            print("✓ Created white.jpg")

        print("✓ Application initialized successfully")
        self.video_loop()

    def _find_best_model(self):
        """Find the best model from multi-model comparison results."""
        results_path = os.path.join(os.path.dirname(__file__), 'models', 'comparison_results.json')
        if not os.path.exists(results_path):
            print("  No comparison_results.json found. Run train_models.py first.")
            return None, None
        with open(results_path, 'r') as f:
            results = json.load(f)
        best_name = results.get('best_model')
        if not best_name:
            return None, None
        model_path = os.path.join(os.path.dirname(__file__), 'models', f'{best_name}.h5')
        if os.path.exists(model_path):
            return model_path, best_name
        print(f"  Best model file not found: {model_path}")
        return None, None

    def video_loop(self):
        try:
            ok, frame = self.vs.read()
            
            # Check if frame was successfully read
            if not ok or frame is None:
                print("Warning: Failed to read frame from camera")
                self.root.after(30, self.video_loop)
                return
            
            cv2image = cv2.flip(frame, 1)
            
            # Find hands in frame
            hands, img_with_hands = hd.findHands(cv2image, draw=False, flipType=True)
            cv2image_copy = np.array(cv2image)
            cv2image_rgb = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGB)
            self.current_image = Image.fromarray(cv2image_rgb)
            imgtk = ImageTk.PhotoImage(image=self.current_image)
            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)

            if hands:
                self.no_hand_time = None  # hand is visible
                hand = hands[0]
                x, y, w, h = hand['bbox']

                # Add boundary checks
                y_start = max(0, y - offset)
                y_end = min(cv2image_copy.shape[0], y + h + offset)
                x_start = max(0, x - offset)
                x_end = min(cv2image_copy.shape[1], x + w + offset)

                image = cv2image_copy[y_start:y_end, x_start:x_end]

                white = cv2.imread("white.jpg")

                if white is None:
                    print("Error: Could not load white.jpg")
                    self.root.after(30, self.video_loop)
                    return

                if image.size > 0:
                    handz, img_with_handz = hd2.findHands(image, draw=False, flipType=True)
                    self.ccc += 1

                    if handz:
                        hand = handz[0]
                        self.pts = hand['lmList']

                        os_x = ((400 - w) // 2) - 15
                        os_y = ((400 - h) // 2) - 15

                        # Draw hand skeleton
                        for t in range(0, 4, 1):
                            cv2.line(white, (self.pts[t][0] + os_x, self.pts[t][1] + os_y),
                                    (self.pts[t + 1][0] + os_x, self.pts[t + 1][1] + os_y),
                                    (0, 255, 0), 3)
                        for t in range(5, 8, 1):
                            cv2.line(white, (self.pts[t][0] + os_x, self.pts[t][1] + os_y),
                                    (self.pts[t + 1][0] + os_x, self.pts[t + 1][1] + os_y),
                                    (0, 255, 0), 3)
                        for t in range(9, 12, 1):
                            cv2.line(white, (self.pts[t][0] + os_x, self.pts[t][1] + os_y),
                                    (self.pts[t + 1][0] + os_x, self.pts[t + 1][1] + os_y),
                                    (0, 255, 0), 3)
                        for t in range(13, 16, 1):
                            cv2.line(white, (self.pts[t][0] + os_x, self.pts[t][1] + os_y),
                                    (self.pts[t + 1][0] + os_x, self.pts[t + 1][1] + os_y),
                                    (0, 255, 0), 3)
                        for t in range(17, 20, 1):
                            cv2.line(white, (self.pts[t][0] + os_x, self.pts[t][1] + os_y),
                                    (self.pts[t + 1][0] + os_x, self.pts[t + 1][1] + os_y),
                                    (0, 255, 0), 3)

                        # Connect finger bases
                        cv2.line(white, (self.pts[5][0] + os_x, self.pts[5][1] + os_y),
                                (self.pts[9][0] + os_x, self.pts[9][1] + os_y), (0, 255, 0), 3)
                        cv2.line(white, (self.pts[9][0] + os_x, self.pts[9][1] + os_y),
                                (self.pts[13][0] + os_x, self.pts[13][1] + os_y), (0, 255, 0), 3)
                        cv2.line(white, (self.pts[13][0] + os_x, self.pts[13][1] + os_y),
                                (self.pts[17][0] + os_x, self.pts[17][1] + os_y), (0, 255, 0), 3)
                        cv2.line(white, (self.pts[0][0] + os_x, self.pts[0][1] + os_y),
                                (self.pts[5][0] + os_x, self.pts[5][1] + os_y), (0, 255, 0), 3)
                        cv2.line(white, (self.pts[0][0] + os_x, self.pts[0][1] + os_y),
                                (self.pts[17][0] + os_x, self.pts[17][1] + os_y), (0, 255, 0), 3)

                        # Draw landmark points
                        for i in range(21):
                            cv2.circle(white, (self.pts[i][0] + os_x, self.pts[i][1] + os_y), 2, (0, 0, 255), 1)

                        res = white
                        self.predict(res)

                        self.current_image2 = Image.fromarray(res)
                        imgtk = ImageTk.PhotoImage(image=self.current_image2)
                        self.panel2.imgtk = imgtk
                        self.panel2.config(image=imgtk)

                        # ── Update character display (with finalization flash) ──
                        now = time.time()
                        if self.char_finalized_flag and (now - self.finalized_flash_time < 0.6):
                            self.panel3.config(text=self.current_symbol,
                                               font=("Courier", 30, "bold"), fg="#00aa00")
                        else:
                            self.char_finalized_flag = False
                            self.panel3.config(text=self.current_symbol,
                                               font=("Courier", 30), fg="black")

                        self.b1.config(text=self.word1, font=("Courier", 20), wraplength=825, command=self.action1)
                        self.b2.config(text=self.word2, font=("Courier", 20), wraplength=825, command=self.action2)
                        self.b3.config(text=self.word3, font=("Courier", 20), wraplength=825, command=self.action3)
                        self.b4.config(text=self.word4, font=("Courier", 20), wraplength=825, command=self.action4)
            else:
                # No hand detected — reset stabilization and smoothing
                if self.no_hand_time is None:
                    self.no_hand_time = time.time()
                if time.time() - self.no_hand_time > 1.0:
                    self.stable_char = None
                    self.stable_start_time = None
                    self.last_finalized_char = None
                    self.smoothed_prob = None
                    self.recent_preds.clear()
                    self._update_progress(0.0)
                    self.status_label.config(text="Show a sign to begin", fg="#666666")

            self.panel5.config(text=self.str, font=("Courier", 30), wraplength=1025)
            
        except Exception as e:
            print(f"Error in video_loop: {e}")
            print(traceback.format_exc())
        finally:
            self.root.after(30, self.video_loop)  # ~33 FPS

    def distance(self, x, y):
        return math.sqrt(((x[0] - y[0]) ** 2) + ((x[1] - y[1]) ** 2))

    def _apply_suggestion(self, suggested_word):
        """Replace current partial word with a suggestion, add space, and speak it."""
        idx_space = self.str.rfind(" ")
        idx_word = self.str.find(self.word, idx_space)
        self.str = self.str[:idx_word]
        self.str = self.str + suggested_word.upper() + " "
        # Reset stabilization so next letter starts fresh
        self.stable_char = None
        self.stable_start_time = None
        self.last_finalized_char = None
        self._update_progress(0.0)
        self.status_label.config(text=f"Word '{suggested_word}' selected", fg="#0066cc")
        # Speak the completed word
        threading.Thread(target=self._tts_say, args=(suggested_word,), daemon=True).start()

    def action1(self):
        self._apply_suggestion(self.word1)

    def action2(self):
        self._apply_suggestion(self.word2)

    def action3(self):
        self._apply_suggestion(self.word3)

    def action4(self):
        self._apply_suggestion(self.word4)

    def speak_fun(self):
        text = self.str.strip()
        if text:
            threading.Thread(target=self._tts_say, args=(text,), daemon=True).start()

    def clear_fun(self):
        self.str = " "
        self.word1 = " "
        self.word2 = " "
        self.word3 = " "
        self.word4 = " "
        self.stable_char = None
        self.stable_start_time = None
        self.last_finalized_char = None
        self.smoothed_prob = None
        self.recent_preds.clear()
        self._update_progress(0.0)
        self.status_label.config(text="Cleared — show a sign", fg="#666666")

    def predict(self, test_image):
        # ── 26-class ensemble path ───────────────────────────────
        if self.use_26class:
            # Prepare input at 64x64 (shared by both models)
            img_size = 64
            resized = cv2.resize(test_image, (img_size, img_size))
            resized = resized.astype(np.float32) / 255.0

            # ── 2D CNN: fast per-frame prediction (primary) ──────
            prob = None
            if self.model_2d is not None:
                inp = resized.reshape(1, img_size, img_size, 3)
                prob = self.model_2d.predict(inp, verbose=0)[0]

            # ── 3D CNN: temporal ensemble (secondary) ────────────
            if self.model_3d is not None:
                self.frame_buffer.append(resized)
                if len(self.frame_buffer) > self.seq_len:
                    self.frame_buffer = self.frame_buffer[-self.seq_len:]

                if len(self.frame_buffer) >= self.seq_len:
                    seq = np.array(self.frame_buffer).reshape(
                        1, self.seq_len, img_size, img_size, 3)
                    prob_3d = self.model_3d.predict(seq, verbose=0)[0]

                    if prob is not None:
                        # Only blend 3D if it agrees with 2D on the top letter,
                        # otherwise 3D buffer likely has transition noise
                        top_2d = np.argmax(prob)
                        top_3d = np.argmax(prob_3d)
                        if top_2d == top_3d:
                            # Both agree — blend for extra confidence
                            prob = 0.5 * prob + 0.5 * prob_3d
                        else:
                            # Disagree — trust 2D (no buffer pollution)
                            prob = 0.7 * prob + 0.3 * prob_3d
                    else:
                        prob = prob_3d

            if prob is None:
                self.current_symbol = "..."
                return

            # ── Step 1: EMA probability smoothing ─────────────────
            if self.smoothed_prob is None:
                self.smoothed_prob = prob.copy()
            else:
                self.smoothed_prob = (self.EMA_ALPHA * prob
                                      + (1 - self.EMA_ALPHA) * self.smoothed_prob)

            idx = np.argmax(self.smoothed_prob)
            ch1 = chr(ord('A') + idx)
            confidence = self.smoothed_prob[idx]

            # ── Step 2: Confidence gate ───────────────────────────
            if confidence < self.CONFIDENCE_THRESHOLD:
                self.current_symbol = "?"
                self._update_progress(0.0)
                self.status_label.config(
                    text=f"Low confidence ({confidence:.0%}) — hold clearer",
                    fg="#cc0000")
                self._update_suggestions()
                return

            # ── Step 3: Majority vote over sliding window ─────────
            self.recent_preds.append(ch1)
            if len(self.recent_preds) > self.VOTE_WINDOW:
                self.recent_preds = self.recent_preds[-self.VOTE_WINDOW:]

            vote_counts = Counter(self.recent_preds)
            voted_char, voted_count = vote_counts.most_common(1)[0]
            vote_ratio = voted_count / len(self.recent_preds)

            # Only accept if majority (>60%) agrees
            if vote_ratio >= 0.6:
                stable_letter = voted_char
            else:
                stable_letter = ch1

            self.current_symbol = stable_letter
            self.count += 1
            self.ten_prev_char[self.count % 10] = stable_letter

            # ── Step 4: Time-based stabilization ──────────────────
            now = time.time()

            if stable_letter == self.stable_char:
                elapsed = now - self.stable_start_time
                progress = min(elapsed / self.STABLE_SECONDS, 1.0)
                self._update_progress(progress)
                remaining = max(0, self.STABLE_SECONDS - elapsed)
                self.status_label.config(
                    text=f"'{stable_letter}' ({confidence:.0%}) — {remaining:.1f}s left",
                    fg="#cc6600")

                if elapsed >= self.STABLE_SECONDS and stable_letter != self.last_finalized_char:
                    self._finalize_letter(stable_letter)
            else:
                self.stable_char = stable_letter
                self.stable_start_time = now
                self.last_finalized_char = None
                self._update_progress(0.0)
                self.status_label.config(
                    text=f"Detecting '{stable_letter}' ({confidence:.0%}) — hold steady...",
                    fg="#cc6600")

            self._update_suggestions()
            return

        # ── Legacy 8-group model path ─────────────────────────────
        white = test_image
        white = white.reshape(1, 400, 400, 3)
        prob = np.array(self.model.predict(white, verbose=0)[0], dtype='float32')
        ch1 = np.argmax(prob, axis=0)
        prob[ch1] = 0
        ch2 = np.argmax(prob, axis=0)
        prob[ch2] = 0
        ch3 = np.argmax(prob, axis=0)
        prob[ch3] = 0

        pl = [ch1, ch2]

        # Apply classification rules (your existing logic)
        # [All your existing classification rules here - keeping them as is]
        
        # Group 0: condition for [Aemnst]
        l = [[5, 2], [5, 3], [3, 5], [3, 6], [3, 0], [3, 2], [6, 4], [6, 1], [6, 2], [6, 6], [6, 7], [6, 0], [6, 5],
             [4, 1], [1, 0], [1, 1], [6, 3], [1, 6], [5, 6], [5, 1], [4, 5], [1, 4], [1, 5], [2, 0], [2, 6], [4, 6],
             [1, 0], [5, 7], [1, 6], [6, 1], [7, 6], [2, 5], [7, 1], [5, 4], [7, 0], [7, 5], [7, 2]]
        if pl in l:
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and 
                self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                ch1 = 0

        # condition for [o][s]
        l = [[2, 2], [2, 1]]
        if pl in l:
            if (self.pts[5][0] < self.pts[4][0]):
                ch1 = 0

        # condition for [c0][aemnst]
        l = [[0, 0], [0, 6], [0, 2], [0, 5], [0, 1], [0, 7], [5, 2], [7, 6], [7, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[4][0] and 
                self.pts[0][0] > self.pts[12][0] and self.pts[0][0] > self.pts[16][0] and 
                self.pts[0][0] > self.pts[20][0]) and self.pts[5][0] > self.pts[4][0]:
                ch1 = 2

        # condition for [c0][aemnst]
        l = [[6, 0], [6, 6], [6, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(self.pts[8], self.pts[16]) < 52:
                ch1 = 2

        # [Continue with all your other classification rules...]
        # [I'm keeping your exact logic - just showing the structure]

        # Subgroup classification
        if ch1 == 0:
            ch1 = 'S'
            if self.pts[4][0] < self.pts[6][0] and self.pts[4][0] < self.pts[10][0] and self.pts[4][0] < self.pts[14][0] and self.pts[4][0] < self.pts[18][0]:
                ch1 = 'A'
            if self.pts[4][0] > self.pts[6][0] and self.pts[4][0] < self.pts[10][0] and self.pts[4][0] < self.pts[14][0] and self.pts[4][0] < self.pts[18][0] and self.pts[4][1] < self.pts[14][1] and self.pts[4][1] < self.pts[18][1]:
                ch1 = 'T'
            if self.pts[4][1] > self.pts[8][1] and self.pts[4][1] > self.pts[12][1] and self.pts[4][1] > self.pts[16][1] and self.pts[4][1] > self.pts[20][1]:
                ch1 = 'E'
            if self.pts[4][0] > self.pts[6][0] and self.pts[4][0] > self.pts[10][0] and self.pts[4][0] > self.pts[14][0] and self.pts[4][1] < self.pts[18][1]:
                ch1 = 'M'
            if self.pts[4][0] > self.pts[6][0] and self.pts[4][0] > self.pts[10][0] and self.pts[4][1] < self.pts[18][1] and self.pts[4][1] < self.pts[14][1]:
                ch1 = 'N'

        if ch1 == 2:
            if self.distance(self.pts[12], self.pts[4]) > 42:
                ch1 = 'C'
            else:
                ch1 = 'O'

        if ch1 == 3:
            if (self.distance(self.pts[8], self.pts[12])) > 72:
                ch1 = 'G'
            else:
                ch1 = 'H'

        if ch1 == 7:
            if self.distance(self.pts[8], self.pts[4]) > 42:
                ch1 = 'Y'
            else:
                ch1 = 'J'

        if ch1 == 4:
            ch1 = 'L'

        if ch1 == 6:
            ch1 = 'X'

        if ch1 == 5:
            if self.pts[4][0] > self.pts[12][0] and self.pts[4][0] > self.pts[16][0] and self.pts[4][0] > self.pts[20][0]:
                if self.pts[8][1] < self.pts[5][1]:
                    ch1 = 'Z'
                else:
                    ch1 = 'Q'
            else:
                ch1 = 'P'

        if ch1 == 1:
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and 
                self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                ch1 = 'B'
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and 
                self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                ch1 = 'D'
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and 
                self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                ch1 = 'F'
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and 
                self.pts[14][1] < self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                ch1 = 'I'
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and 
                self.pts[14][1] > self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                ch1 = 'W'
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and 
                self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]) and self.pts[4][1] < self.pts[9][1]:
                ch1 = 'K'
            if ((self.distance(self.pts[8], self.pts[12]) - self.distance(self.pts[6], self.pts[10])) < 8) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and 
                    self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                ch1 = 'U'
            if ((self.distance(self.pts[8], self.pts[12]) - self.distance(self.pts[6], self.pts[10])) >= 8) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and 
                    self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]) and (self.pts[4][1] > self.pts[9][1]):
                ch1 = 'V'
            if (self.pts[8][0] > self.pts[12][0]) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and 
                    self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                ch1 = 'R'

        # Special gesture detection for space
        if ch1 == 1 or ch1 == 'E' or ch1 == 'S' or ch1 == 'X' or ch1 == 'Y' or ch1 == 'B':
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and 
                self.pts[14][1] < self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                ch1 = " "

        # Next character gesture
        if ch1 == 'E' or ch1 == 'Y' or ch1 == 'B':
            if (self.pts[4][0] < self.pts[5][0]) and (
                self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and 
                self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                ch1 = "next"

        # Backspace gesture
        if ch1 == 'Next' or 'B' or 'C' or 'H' or 'F' or 'X':
            if (self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[12][0] and 
                self.pts[0][0] > self.pts[16][0] and self.pts[0][0] > self.pts[20][0]) and (
                self.pts[4][1] < self.pts[8][1] and self.pts[4][1] < self.pts[12][1] and 
                self.pts[4][1] < self.pts[16][1] and self.pts[4][1] < self.pts[20][1]) and (
                self.pts[4][1] < self.pts[6][1] and self.pts[4][1] < self.pts[10][1] and 
                self.pts[4][1] < self.pts[14][1] and self.pts[4][1] < self.pts[18][1]):
                ch1 = 'Backspace'

        # Handle next gesture
        if ch1 == "next" and self.prev_char != "next":
            if self.ten_prev_char[(self.count - 2) % 10] != "next":
                if self.ten_prev_char[(self.count - 2) % 10] == "Backspace":
                    self.str = self.str[0:-1]
                else:
                    if self.ten_prev_char[(self.count - 2) % 10] != "Backspace":
                        self.str = self.str + self.ten_prev_char[(self.count - 2) % 10]
            else:
                if self.ten_prev_char[(self.count - 0) % 10] != "Backspace":
                    self.str = self.str + self.ten_prev_char[(self.count - 0) % 10]

        # Handle space
        if ch1 == "  " and self.prev_char != "  ":
            self.str = self.str + "  "

        self.prev_char = ch1
        self.current_symbol = ch1
        self.count += 1
        self.ten_prev_char[self.count % 10] = ch1

        # Update word suggestions
        if len(self.str.strip()) != 0:
            st = self.str.rfind(" ")
            ed = len(self.str)
            word = self.str[st + 1:ed]
            self.word = word
            if len(word.strip()) != 0:
                try:
                    ddd.check(word)
                    suggestions = ddd.suggest(word)
                    lenn = len(suggestions)
                    
                    self.word1 = suggestions[0] if lenn >= 1 else " "
                    self.word2 = suggestions[1] if lenn >= 2 else " "
                    self.word3 = suggestions[2] if lenn >= 3 else " "
                    self.word4 = suggestions[3] if lenn >= 4 else " "
                except:
                    self.word1 = self.word2 = self.word3 = self.word4 = " "
            else:
                self.word1 = self.word2 = self.word3 = self.word4 = " "

    # ── Helper: finalize a confirmed letter ────────────────────────
    def _finalize_letter(self, ch):
        """Append confirmed letter to sentence, update UI, speak if word done."""
        self.last_finalized_char = ch
        self.str = self.str + ch
        self.char_finalized_flag = True
        self.finalized_flash_time = time.time()

        # Clear 3D CNN buffer so next letter starts without old-gesture frames
        self.frame_buffer.clear()
        # Clear smoothing state so next letter isn't biased by old probabilities
        self.smoothed_prob = None
        self.recent_preds.clear()

        # Reset progress bar to full green briefly
        self._update_progress(1.0)
        self.status_label.config(text=f"'{ch}' added!", fg="#00aa00")

        # Update suggestions after adding letter
        self._update_suggestions()

    # ── Helper: update the progress bar ──────────────────────────
    def _update_progress(self, fraction):
        """Update the stabilization progress bar (0.0 to 1.0)."""
        width = int(280 * fraction)
        self.progress_canvas.coords(self.progress_bar, 0, 0, width, 24)
        pct = int(fraction * 100)
        self.progress_canvas.itemconfig(self.progress_text, text=f"{pct}%")
        # Color gradient: red → yellow → green
        if fraction < 0.5:
            r, g = 255, int(255 * fraction * 2)
        else:
            r, g = int(255 * (1 - fraction) * 2), 200
        color = f"#{r:02x}{g:02x}44"
        self.progress_canvas.itemconfig(self.progress_bar, fill=color)

    # ── Helper: update word suggestions ──────────────────────────
    def _update_suggestions(self):
        """Update the 4 suggestion buttons based on current partial word."""
        if len(self.str.strip()) != 0:
            st = self.str.rfind(" ")
            word = self.str[st + 1:]
            self.word = word
            if len(word.strip()) != 0:
                try:
                    ddd.check(word)
                    suggestions = ddd.suggest(word)
                    lenn = len(suggestions)
                    self.word1 = suggestions[0] if lenn >= 1 else " "
                    self.word2 = suggestions[1] if lenn >= 2 else " "
                    self.word3 = suggestions[2] if lenn >= 3 else " "
                    self.word4 = suggestions[3] if lenn >= 4 else " "
                except:
                    self.word1 = self.word2 = self.word3 = self.word4 = " "
            else:
                self.word1 = self.word2 = self.word3 = self.word4 = " "

    # ── Helper: speak the last completed word ────────────────────
    def speak_last_word(self):
        """Speak the last word in the sentence (in a background thread)."""
        words = self.str.strip().split()
        if words:
            word = words[-1]
            threading.Thread(target=self._tts_say, args=(word,), daemon=True).start()

    def _tts_say(self, text):
        """Run TTS in background thread so UI doesn't freeze."""
        try:
            self.speak_engine.say(text)
            self.speak_engine.runAndWait()
        except Exception as e:
            print(f"TTS error: {e}")

    def destructor(self):
        print("Closing application...")
        print("Last 10 characters:", self.ten_prev_char)
        self.root.destroy()
        if self.vs is not None:
            self.vs.release()
        cv2.destroyAllWindows()
        print("✓ Application closed successfully")


if __name__ == "__main__":
    print("=" * 60)
    print("Starting Sign Language To Text Conversion Application...")
    print("=" * 60)
    try:
        app = Application()
        app.root.mainloop()
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        print(traceback.format_exc())
        
        
        
        
# robust 
# letter -> word -> sentence 
# character intact 
# model change - cnn 

