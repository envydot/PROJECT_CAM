"""
06_drawing_board.py
----------------------
Draw in the air with your index finger.

Modes:
    Index finger only up          = DRAW
    Index + middle finger up      = SELECTION mode (move over the color bar to pick a color, or the eraser)
    All fingers down / other      = nothing happens (pen up)

Controls:
    c = clear the whole canvas
    q = quit

Run from the project root:
    python apps/06_drawing_board.py
"""

import cv2
import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "modules"))
from HandTrackingModule import HandDetector

cam_width, cam_height = 1280, 720
cap = cv2.VideoCapture(0)
cap.set(3, cam_width)
cap.set(4, cam_height)

detector = HandDetector(max_hands=1, detection_con=0.8)

# Persistent canvas the same size as the camera frame (drawing accumulates here)
canvas = np.zeros((cam_height, cam_width, 3), np.uint8)

# Color palette: (name, BGR color, x-range on the top bar)
colors = [
    ("Red", (0, 0, 255), (0, 150)),
    ("Green", (0, 255, 0), (150, 300)),
    ("Blue", (255, 0, 0), (300, 450)),
    ("Yellow", (0, 255, 255), (450, 600)),
    ("Eraser", (0, 0, 0), (600, 750)),
]
bar_height = 80
draw_color = (0, 0, 255)  # default red
brush_thickness = 10
eraser_thickness = 60

prev_x, prev_y = 0, 0

while True:
    success, img = cap.read()
    if not success:
        print("Could not read from webcam.")
        break
    img = cv2.flip(img, 1)  # mirror so drawing feels natural

    img = detector.find_hands(img)
    landmark_list = detector.find_position(img, draw=False)

    # --- Draw the color selection bar at the top ---
    for name, color, (x1, x2) in colors:
        cv2.rectangle(img, (x1, 0), (x2, bar_height), color if color != (0, 0, 0) else (50, 50, 50), cv2.FILLED)
        cv2.putText(img, name, (x1 + 10, 50), cv2.FONT_HERSHEY_PLAIN, 1.5,
                    (255, 255, 255) if color != (0, 255, 255) else (0, 0, 0), 2)

    if len(landmark_list) != 0:
        x1, y1 = landmark_list[8][1], landmark_list[8][2]   # index tip
        x2, y2 = landmark_list[12][1], landmark_list[12][2]  # middle tip
        fingers = detector.fingers_up()

        # --- SELECTION MODE: index + middle up ---
        if fingers[1] == 1 and fingers[2] == 1:
            prev_x, prev_y = 0, 0  # reset so we don't draw a line when switching back to draw mode
            if y1 < bar_height:
                for name, color, (cx1, cx2) in colors:
                    if cx1 < x1 < cx2:
                        draw_color = color
            cv2.rectangle(img, (x1 - 10, y1 - 15), (x2 + 10, y2 + 15), draw_color, cv2.FILLED)

        # --- DRAW MODE: only index up ---
        elif fingers[1] == 1 and fingers[2] == 0:
            cv2.circle(img, (x1, y1), 8, draw_color, cv2.FILLED)
            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = x1, y1

            thickness = eraser_thickness if draw_color == (0, 0, 0) else brush_thickness
            cv2.line(canvas, (prev_x, prev_y), (x1, y1), draw_color, thickness)
            prev_x, prev_y = x1, y1
        else:
            prev_x, prev_y = 0, 0

    # --- Merge canvas onto the live camera feed ---
    # Convert canvas to grayscale to build a mask of "drawn" pixels
    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, inv_mask = cv2.threshold(gray_canvas, 20, 255, cv2.THRESH_BINARY_INV)
    inv_mask = cv2.cvtColor(inv_mask, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, inv_mask)
    img = cv2.bitwise_or(img, canvas)

    cv2.imshow("Virtual Drawing Board", img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("c"):
        canvas = np.zeros((cam_height, cam_width, 3), np.uint8)

cap.release()
cv2.destroyAllWindows()