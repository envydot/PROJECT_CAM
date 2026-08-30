"""
01_finger_counter.py
---------------------
The simplest app: shows a live webcam feed, detects your hand,
and displays how many fingers are up.

Run this from the `pythoncv` folder root, NOT from inside `apps/`,
so the import path below works correctly.
"""

import cv2
import sys
import os

# Allows this script to find modules/HandTrackingModule.py
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "modules"))

from HandTrackingModule import HandDetector

# --- Setup ---
cap = cv2.VideoCapture(0)  # 0 = default webcam
cap.set(3, 1280)  # width
cap.set(4, 720)   # height

detector = HandDetector(detection_con=0.75, max_hands=1)

while True:
    success, img = cap.read()
    if not success:
        print("Could not read from webcam. Check that it's connected and not used by another app.")
        break

    img = detector.find_hands(img)
    landmark_list = detector.find_position(img, draw=False)

    if len(landmark_list) != 0:
        fingers = detector.fingers_up()
        count = fingers.count(1)

        # Draw a background box + big number
        cv2.rectangle(img, (20, 20), (170, 140), (0, 0, 0), cv2.FILLED)
        cv2.putText(img, str(count), (45, 115), cv2.FONT_HERSHEY_PLAIN,
                    8, (0, 255, 0), 8)

    cv2.imshow("Finger Counter", img)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()