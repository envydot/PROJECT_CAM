"""
02_virtual_mouse.py
---------------------
Move your mouse cursor with your index finger.
Pinch (thumb tip touching index tip) = left click.

Run from the project root:
    python apps/02_virtual_mouse.py
"""

import cv2
import sys
import os
import time
import numpy as np
import pyautogui

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "modules"))
from HandTrackingModule import HandDetector

# --- Settings ---
cam_width, cam_height = 640, 480
frame_reduction = 100     # margin so you don't need to reach screen edges
smoothening = 5           # higher = smoother but more delayed cursor
click_distance = 35       # pixel distance between thumb+index = "pinch"

# --- Setup ---
cap = cv2.VideoCapture(0)
cap.set(3, cam_width)
cap.set(4, cam_height)

detector = HandDetector(max_hands=1, detection_con=0.8)
screen_width, screen_height = pyautogui.size()

prev_x, prev_y = 0, 0
curr_x, curr_y = 0, 0

# Safety: prevent pyautogui's fail-safe corner trigger from crashing the app
pyautogui.FAILSAFE = False

while True:
    success, img = cap.read()
    if not success:
        print("Could not read from webcam.")
        break

    img = detector.find_hands(img)
    landmark_list = detector.find_position(img, draw=False)

    # Draw the "active zone" rectangle for reference
    cv2.rectangle(img, (frame_reduction, frame_reduction),
                  (cam_width - frame_reduction, cam_height - frame_reduction),
                  (255, 0, 255), 2)

    if len(landmark_list) != 0:
        fingers = detector.fingers_up()

        # Index fingertip position
        x1, y1 = landmark_list[8][1], landmark_list[8][2]

        # --- MOVE MODE: only index finger up ---
        if fingers[1] == 1 and fingers[2] == 0:
            # Map camera coordinates -> screen coordinates
            screen_x = np.interp(x1, (frame_reduction, cam_width - frame_reduction), (0, screen_width))
            screen_y = np.interp(y1, (frame_reduction, cam_height - frame_reduction), (0, screen_height))

            # Smoothen movement so cursor doesn't jitter
            curr_x = prev_x + (screen_x - prev_x) / smoothening
            curr_y = prev_y + (screen_y - prev_y) / smoothening

            pyautogui.moveTo(screen_width - curr_x, curr_y)  # flip x for mirror effect
            cv2.circle(img, (x1, y1), 12, (255, 0, 255), cv2.FILLED)
            prev_x, prev_y = curr_x, curr_y

        # --- CLICK MODE: check pinch distance between thumb(4) and index(8) ---
        if fingers[1] == 1 and fingers[0] == 1:
            length, img, line_info = detector.find_distance(4, 8, img)
            if length < click_distance:
                cv2.circle(img, (line_info[4], line_info[5]), 15, (0, 255, 0), cv2.FILLED)
                pyautogui.click()
                time.sleep(0.2)  # prevents multiple clicks per pinch

    cv2.imshow("Virtual Mouse", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()