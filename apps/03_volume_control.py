"""
03_volume_control.py
----------------------
Control system volume with your hand.
Distance between thumb tip and index tip -> volume level.
Close together = quiet/mute, far apart = loud.

Windows only (uses pycaw to access system audio).

Run from the project root:
    python apps/03_volume_control.py
"""

import cv2
import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "modules"))
from HandTrackingModule import HandDetector

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# --- Setup system audio control ---
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume_ctrl = cast(interface, POINTER(IAudioEndpointVolume))

vol_range = volume_ctrl.GetVolumeRange()
min_vol, max_vol = vol_range[0], vol_range[1]  # usually around -65 to 0 (in dB)

# --- Setup camera + hand detector ---
cam_width, cam_height = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, cam_width)
cap.set(4, cam_height)

detector = HandDetector(max_hands=1, detection_con=0.8)

# Distance range from your hand (measured in pixels) that maps to volume
min_hand_dist = 25
max_hand_dist = 200

while True:
    success, img = cap.read()
    if not success:
        print("Could not read from webcam.")
        break

    img = detector.find_hands(img)
    landmark_list = detector.find_position(img, draw=False)

    if len(landmark_list) != 0:
        # Only adjust volume when thumb + index are up (avoids accidental changes)
        fingers = detector.fingers_up()

        if fingers[0] == 1 and fingers[1] == 1:
            length, img, line_info = detector.find_distance(4, 8, img)

            # Map hand distance -> volume range (in dB, as pycaw expects)
            vol = np.interp(length, [min_hand_dist, max_hand_dist], [min_vol, max_vol])
            vol_bar = np.interp(length, [min_hand_dist, max_hand_dist], [400, 150])
            vol_percent = np.interp(length, [min_hand_dist, max_hand_dist], [0, 100])

            volume_ctrl.SetMasterVolumeLevel(vol, None)

            # Turn the connecting line green when very close (near-mute)
            if length < min_hand_dist + 5:
                cv2.circle(img, (line_info[4], line_info[5]), 15, (0, 255, 0), cv2.FILLED)

            # --- Draw volume bar UI ---
            cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
            cv2.rectangle(img, (50, int(vol_bar)), (85, 400), (255, 0, 0), cv2.FILLED)
            cv2.putText(img, f'{int(vol_percent)} %', (40, 430),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    cv2.imshow("Volume Control", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()