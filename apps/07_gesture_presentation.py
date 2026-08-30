"""
07_gesture_presentation.py
-----------------------------
Control slides (or any app) with hand gestures - no clicker needed.

Gestures (hand must be raised in the top/bottom "trigger zone" show on screen):
    Thumb only up            = PREVIOUS slide (left arrow key)
    Pinky only up             = NEXT slide (right arrow key)

Works with ANY presentation software that responds to arrow keys
(PowerPoint, Google Slides, PDF viewers in presentation mode, etc).
Just open your presentation in fullscreen BEFORE running this, then
alt-tab is not needed - pyautogui sends keys to whatever is focused.

Run from the project root:
    python apps/07_gesture_presentation.py
"""

import cv2
import sys
import os
import time
import pyautogui

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "modules"))
from HandTrackingModule import HandDetector

cam_width, cam_height = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, cam_width)
cap.set(4, cam_height)

detector = HandDetector(max_hands=1, detection_con=0.8)

gesture_cooldown = 1.2  # seconds between slide changes, so one gesture doesn't fire repeatedly
last_action_time = 0

while True:
    success, img = cap.read()
    if not success:
        print("Could not read from webcam.")
        break
    img = cv2.flip(img, 1)

    img = detector.find_hands(img, draw=True)
    landmark_list = detector.find_position(img, draw=False)

    action_text = ""
    if len(landmark_list) != 0:
        fingers = detector.fingers_up()  # [thumb, index, middle, ring, pinky]
        now = time.time()

        if now - last_action_time > gesture_cooldown:
            # Thumb only -> previous
            if fingers == [1, 0, 0, 0, 0]:
                pyautogui.press("left")
                action_text = "<< PREVIOUS"
                last_action_time = now

            # Pinky only -> next
            elif fingers == [0, 0, 0, 0, 1]:
                pyautogui.press("right")
                action_text = "NEXT >>"
                last_action_time = now

    if action_text:
        cv2.putText(img, action_text, (150, 240), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 4)

    cv2.putText(img, "Thumb=Prev  Pinky=Next  q=quit", (10, cam_height - 20),
                cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)

    cv2.imshow("Gesture Presentation Control", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()