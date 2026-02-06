import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import os
import traceback

BASE_DATASET = "/home/veeransh/Desktop/Sign-Language-To-Text-and-Speech-Conversion/AtoZ_3.1"
SRC_DIR = os.path.join(BASE_DATASET, "AtoZ_3.0")
DEST_DIR = os.path.join(BASE_DATASET, "AtoZ_3.1")


for ch in range(ord('A'), ord('Z') + 1):
    os.makedirs(os.path.join(DEST_DIR, chr(ch)), exist_ok=True)


capture = cv2.VideoCapture(0)
if not capture.isOpened():
    raise RuntimeError("❌ Cannot open webcam. Is it already in use?")

hd = HandDetector(maxHands=1)
hd2 = HandDetector(maxHands=1)

#
c_dir = 'A'
count = len(os.listdir(os.path.join(SRC_DIR, c_dir)))

offset = 15
step = 1
flag = False
suv = 0


WHITE_SIZE = 400

while True:
    try:
        success, frame = capture.read()
        if not success:
            print("⚠️ Failed to read frame from camera")
            continue

        frame = cv2.flip(frame, 1)
        hands = hd.findHands(frame, draw=False, flipType=True)

        white = np.ones((WHITE_SIZE, WHITE_SIZE, 3), np.uint8) * 255

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            image = frame[
                max(0, y - offset): y + h + offset,
                max(0, x - offset): x + w + offset
            ]

            handz, _ = hd2.findHands(image, draw=False, flipType=True)

            if handz:
                hand = handz[0]
                pts = hand['lmList']

                os_x = ((WHITE_SIZE - w) // 2) - 15
                os_y = ((WHITE_SIZE - h) // 2) - 15

                # Draw fingers
                connections = [
                    (0, 5), (5, 9), (9, 13), (13, 17), (0, 17),
                    (0, 1), (1, 2), (2, 3), (3, 4),
                    (5, 6), (6, 7), (7, 8),
                    (9, 10), (10, 11), (11, 12),
                    (13, 14), (14, 15), (15, 16),
                    (17, 18), (18, 19), (19, 20)
                ]

                for i, j in connections:
                    cv2.line(
                        white,
                        (pts[i][0] + os_x, pts[i][1] + os_y),
                        (pts[j][0] + os_x, pts[j][1] + os_y),
                        (0, 255, 0),
                        3
                    )

                for i in range(21):
                    cv2.circle(
                        white,
                        (pts[i][0] + os_x, pts[i][1] + os_y),
                        2,
                        (0, 0, 255),
                        1
                    )

                skeleton1 = np.array(white)
                cv2.imshow("Skeleton", skeleton1)

        # Overlay info
        frame = cv2.putText(
            frame,
            f"dir={c_dir}  count={count}",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            1,
            cv2.LINE_AA
        )

        cv2.imshow("Frame", frame)

        interrupt = cv2.waitKey(1) & 0xFF

        # ESC
        if interrupt == 27:
            break

        # Next letter
        if interrupt == ord('n'):
            c_dir = chr(ord(c_dir) + 1)
            if c_dir > 'Z':
                c_dir = 'A'
            flag = False
            count = len(os.listdir(os.path.join(SRC_DIR, c_dir)))

        # Toggle capture
        if interrupt == ord('a'):
            flag = not flag
            suv = 0

        print("=====", flag)

        if flag:
            if suv >= 180:
                flag = False

            if step % 3 == 0:
                save_path = os.path.join(
                    DEST_DIR,
                    c_dir,
                    f"{count}.jpg"
                )
                cv2.imwrite(save_path, skeleton1)
                count += 1
                suv += 1

            step += 1

    except Exception:
        print("❌ ERROR:\n", traceback.format_exc())

# ===================== CLEANUP =====================
capture.release()
cv2.destroyAllWindows()
