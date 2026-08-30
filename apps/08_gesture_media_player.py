"""
08_gesture_media_player.py
------------------------------
Control media playback (Spotify, YouTube, VLC, etc.) with hand gestures.
Works with anything that responds to standard media keys.

Gestures:
    Open palm (all 5 up)              = Play / Pause
    Fist (0 fingers up)                = Stop (mapped to pause too, most apps don't separate stop)
    Thumb only up                      = Volume Up
    Pinky only up                      = Volume Down
    Index only up                      = Next track
    Index + pinky up ("rock sign")     = Previous track

Run from the project root:
    python apps/08_gesture_media_player.py
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

cooldown = 1.0
last_action_time = 0


def match(fingers, pattern):
    return fingers == pattern


while True:
    success, img = cap.read()
    if not success:
        print("Could not read from webcam.")
        break
    img = cv2.flip(img, 1)

    img = detector.find_hands(img)
    landmark_list = detector.find_position(img, draw=False)

    action_text = ""
    if len(landmark_list) != 0:
        fingers = detector.fingers_up()
        now = time.time()

        if now - last_action_time > cooldown:
            if match(fingers, [1, 1, 1, 1, 1]):
                pyautogui.press("playpause")
                action_text = "PLAY / PAUSE"
                last_action_time = now
            elif match(fingers, [0, 0, 0, 0, 0]):
                pyautogui.press("playpause")
                action_text = "PLAY / PAUSE"
                last_action_time = now
            elif match(fingers, [1, 0, 0, 0, 0]):
                pyautogui.press("volumeup")
                action_text = "VOLUME UP"
                last_action_time = now
            elif match(fingers, [0, 0, 0, 0, 1]):
                pyautogui.press("volumedown")
                action_text = "VOLUME DOWN"
                last_action_time = now
            elif match(fingers, [0, 1, 0, 0, 0]):
                pyautogui.press("nexttrack")
                action_text = "NEXT TRACK"
                last_action_time = now
            elif match(fingers, [0, 1, 0, 0, 1]):
                pyautogui.press("prevtrack")
                action_text = "PREVIOUS TRACK"
                last_action_time = now

    if action_text:
        cv2.putText(img, action_text, (60, 240), cv2.FONT_HERSHEY_PLAIN, 2.5, (0, 255, 0), 3)

    cv2.putText(img, "Palm/Fist=Play-Pause  Thumb/Pinky=Vol  Index=Next  Rock-sign=Prev",
                (10, cam_height - 15), cv2.FONT_HERSHEY_PLAIN, 1.1, (255, 255, 255), 1)

    cv2.imshow("Gesture Media Player", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()