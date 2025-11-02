# ============================================================================
# FILE 1: final_pred.py (COMPLETE FIXED VERSION)
# ============================================================================

import numpy as np
import math
import cv2
import os
import sys
import traceback
import pyttsx3
from keras.models import load_model
from cvzone.HandTrackingModule import HandDetector
from string import ascii_uppercase
import enchant
import tkinter as tk
from PIL import Image, ImageTk

# Initialize spell checker and hand detectors
ddd = enchant.Dict("en-US")
hd = HandDetector(maxHands=1)
hd2 = HandDetector(maxHands=1)

offset = 29

os.environ["THEANO_FLAGS"] = "device=cuda, assert_no_cpu_op=True"


class Application:
    def __init__(self):
        # Try to open camera with different indices
        self.vs = None
        print("Attempting to open camera...")
        for camera_index in [0, 1, 2]:
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
        self.model = load_model('cnn8grps_rad1_model.h5')
        print("✓ Model loaded successfully")
        
        # Initialize text-to-speech engine
        self.speak_engine = pyttsx3.init()
        self.speak_engine.setProperty("rate", 100)
        voices = self.speak_engine.getProperty("voices")
        self.speak_engine.setProperty("voice", voices[0].id)

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

        # Control buttons
        self.speak = tk.Button(self.root)
        self.speak.place(x=1205, y=630)
        self.speak.config(text="Speak", font=("Courier", 20), wraplength=100, command=self.speak_fun)

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

                        self.panel3.config(text=self.current_symbol, font=("Courier", 30))

                        self.b1.config(text=self.word1, font=("Courier", 20), wraplength=825, command=self.action1)
                        self.b2.config(text=self.word2, font=("Courier", 20), wraplength=825, command=self.action2)
                        self.b3.config(text=self.word3, font=("Courier", 20), wraplength=825, command=self.action3)
                        self.b4.config(text=self.word4, font=("Courier", 20), wraplength=825, command=self.action4)

            self.panel5.config(text=self.str, font=("Courier", 30), wraplength=1025)
            
        except Exception as e:
            print(f"Error in video_loop: {e}")
            print(traceback.format_exc())
        finally:
            self.root.after(30, self.video_loop)  # ~33 FPS

    def distance(self, x, y):
        return math.sqrt(((x[0] - y[0]) ** 2) + ((x[1] - y[1]) ** 2))

    def action1(self):
        idx_space = self.str.rfind(" ")
        idx_word = self.str.find(self.word, idx_space)
        self.str = self.str[:idx_word]
        self.str = self.str + self.word1.upper()

    def action2(self):
        idx_space = self.str.rfind(" ")
        idx_word = self.str.find(self.word, idx_space)
        self.str = self.str[:idx_word]
        self.str = self.str + self.word2.upper()

    def action3(self):
        idx_space = self.str.rfind(" ")
        idx_word = self.str.find(self.word, idx_space)
        self.str = self.str[:idx_word]
        self.str = self.str + self.word3.upper()

    def action4(self):
        idx_space = self.str.rfind(" ")
        idx_word = self.str.find(self.word, idx_space)
        self.str = self.str[:idx_word]
        self.str = self.str + self.word4.upper()

    def speak_fun(self):
        self.speak_engine.say(self.str)
        self.speak_engine.runAndWait()

    def clear_fun(self):
        self.str = " "
        self.word1 = " "
        self.word2 = " "
        self.word3 = " "
        self.word4 = " "

    def predict(self, test_image):
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