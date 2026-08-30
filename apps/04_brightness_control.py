"""
04_brightness_control.py
--------------------------
Control screen brightness with your hand.
Distance between thumb tip and index tip -> brightness level.
Close together = dim, far apart = bright.

Run from the project root:
    python apps/04_brightness_control.py
"""

import cv2
import sys
import os
import numpy as np
import screen_brightness_control as sbc

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "modules"))
from HandTrackingModule import HandDetector

# --- Setup camera + hand detector ---
cam_width, cam_height = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, cam_width)
cap.set(4, cam_height)

detector = HandDetector(max_hands=1, detection_con=0.8)

# Distance range from your hand (pixels) that maps to brightness 0-100
min_hand_dist = 25
max_hand_dist = 200

# Avoid calling sbc.set_brightness() every single frame (it can lag/flicker)
last_brightness = -1

while True:
    success, img = cap.read()
    if not success:
        print("Could not read from webcam.")
        break

    img = detector.find_hands(img)
    landmark_list = detector.find_position(img, draw=False)

    if len(landmark_list) != 0:
        fingers = detector.fingers_up()

        if fingers[0] == 1 and fingers[1] == 1:
            length, img, line_info = detector.find_distance(4, 8, img)

            brightness = int(np.interp(length, [min_hand_dist, max_hand_dist], [0, 100]))
            bar = np.interp(length, [min_hand_dist, max_hand_dist], [400, 150])

            # Only update if brightness changed meaningfully (avoids flicker/lag)
            if abs(brightness - last_brightness) >= 2:
                sbc.set_brightness(brightness)
                last_brightness = brightness

            if length < min_hand_dist + 5:
                cv2.circle(img, (line_info[4], line_info[5]), 15, (0, 255, 0), cv2.FILLED)

            # --- Draw brightness bar UI ---
            cv2.rectangle(img, (50, 150), (85, 400), (0, 215, 255), 3)
            cv2.rectangle(img, (50, int(bar)), (85, 400), (0, 215, 255), cv2.FILLED)
            cv2.putText(img, f'{brightness} %', (40, 430),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 215, 255), 2)

    cv2.imshow("Brightness Control", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()