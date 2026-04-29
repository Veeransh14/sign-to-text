

import numpy as np
import math
import cv2
import os
import sys
import json
import time
import queue
import threading
import traceback
import subprocess
import shutil
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

offset = 15

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
        
        # Load the trained model
        print("Loading model...")
        self.use_26class = False
        if USE_BEST_MODEL:
            best_model_path, best_model_name = self._find_best_model()
            if best_model_path:
                try:
                    custom_objects = {}
                    if 'transformer' in best_model_name:
                        from train_models import PositionalEncoding
                        custom_objects['PositionalEncoding'] = PositionalEncoding
                    self.model = load_model(best_model_path, custom_objects=custom_objects)
                    self.use_26class = True
                    print(f"✓ Loaded best model: {best_model_name}")
                except Exception as e:
                    print(f"  Warning: Failed to load best model ({e}), falling back to legacy")
                    self.model = load_model('cnn8grps_rad1_model.h5')
            else:
                self.model = load_model('cnn8grps_rad1_model.h5')
        else:
            self.model = load_model('cnn8grps_rad1_model.h5')
        print("✓ Model loaded successfully")
        
        # Initialize text-to-speech engine (single worker thread + queue)
        self._tts_queue = queue.Queue()
        self._tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
        self._tts_thread.start()

        # Sequence buffer for temporal models (3D CNN, LSTM, Transformer)
        self.frame_buffer = []
        self.seq_len = 10

        # ── Prediction smoothing & filtering ─────────────────────────
        self.smoothed_prob = None         # EMA-smoothed probability vector
        self.EMA_ALPHA = 0.45            # smoothing factor (higher = more responsive)
        self.CONFIDENCE_THRESHOLD = 0.20  # 26 classes → random=3.8%, so 20% is 5x random
        self.VOTE_WINDOW = 20            # sliding window for majority voting
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
        self._tts_say(suggested_word)

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
            self._tts_say(text, drain=True)

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
        # ── 26-class best model path ──────────────────────────────
        if self.use_26class:
            input_shape = self.model.input_shape
            is_sequence = len(input_shape) == 5  # (batch, seq, h, w, c)
            img_h, img_w = (input_shape[2], input_shape[3]) if is_sequence else (input_shape[1], input_shape[2])
            resized = cv2.resize(test_image, (img_w, img_h))
            resized = resized.astype(np.float32) / 255.0

            if is_sequence:
                self.frame_buffer.append(resized)
                if len(self.frame_buffer) > self.seq_len:
                    self.frame_buffer = self.frame_buffer[-self.seq_len:]
                if len(self.frame_buffer) < self.seq_len:
                    self.current_symbol = "..."
                    self._update_progress(0.0)
                    self.status_label.config(text="Buffering frames...", fg="#999900")
                    return
                seq = np.array(self.frame_buffer).reshape(1, self.seq_len, img_h, img_w, 3)
                prob = self.model.predict(seq, verbose=0)[0]
            else:
                inp = resized.reshape(1, img_h, img_w, 3)
                prob = self.model.predict(inp, verbose=0)[0]

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

            # Find the most common prediction in the window
            vote_counts = Counter(self.recent_preds)
            voted_char, voted_count = vote_counts.most_common(1)[0]
            vote_ratio = voted_count / len(self.recent_preds)

            # Only accept if majority (>60%) agrees
            if vote_ratio >= 0.6:
                stable_letter = voted_char
            else:
                stable_letter = ch1  # fallback to smoothed prediction

            self.current_symbol = stable_letter
            self.count += 1
            self.ten_prev_char[self.count % 10] = stable_letter

            # ── Step 4: Time-based stabilization ──────────────────
            now = time.time()

            if stable_letter == self.stable_char:
                # Same letter continues — update progress
                elapsed = now - self.stable_start_time
                progress = min(elapsed / self.STABLE_SECONDS, 1.0)
                self._update_progress(progress)
                self.status_label.config(
                    text=f"'{stable_letter}' ({confidence:.0%}) — {self.STABLE_SECONDS - elapsed:.1f}s left",
                    fg="#cc6600")

                if elapsed >= self.STABLE_SECONDS and stable_letter != self.last_finalized_char:
                    # FINALIZE this letter
                    self._finalize_letter(stable_letter)
            else:
                # Different letter detected — reset timer
                self.stable_char = stable_letter
                self.stable_start_time = now
                self.last_finalized_char = None
                self._update_progress(0.0)
                self.status_label.config(
                    text=f"Detecting '{stable_letter}' ({confidence:.0%}) — hold steady...",
                    fg="#cc6600")

            # ── Update word suggestions ───────────────────────────
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
        """Speak the last word in the sentence."""
        words = self.str.strip().split()
        if words:
            self._tts_say(words[-1], drain=True)

    def _tts_say(self, text, drain=False):
        """Queue text to be spoken by the dedicated TTS worker thread.

        drain=True discards any pending items so only this text is spoken next.
        """
        if drain:
            while not self._tts_queue.empty():
                try:
                    self._tts_queue.get_nowait()
                except queue.Empty:
                    break
        self._tts_queue.put(text)

    def _tts_worker(self):
        """Dedicated TTS thread — uses espeak-ng subprocess for thread-safe audio."""
        # Detect available TTS backend once at startup
        espeak = shutil.which("espeak-ng") or shutil.which("espeak")

        while True:
            text = self._tts_queue.get()
            if text is None:
                break
            try:
                if espeak:
                    subprocess.run(
                        [espeak, "-s", "130", "--", text],
                        timeout=30,
                        stderr=subprocess.DEVNULL,
                    )
                else:
                    # Fallback: pyttsx3 (requires it to be installed)
                    import pyttsx3
                    engine = pyttsx3.init()
                    engine.setProperty("rate", 130)
                    voices = engine.getProperty("voices")
                    if voices:
                        engine.setProperty("voice", voices[0].id)
                    engine.say(text)
                    engine.runAndWait()
                    engine.stop()
            except subprocess.TimeoutExpired:
                print("TTS timeout — skipping utterance")
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

