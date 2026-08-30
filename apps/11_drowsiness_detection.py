"""
11_drowsiness_detection.py
------------------------------
Detects if the driver's eyes have been closed too long and raises an alert.

Uses Eye Aspect Ratio (EAR): a value computed from 6 points around each eye.
EAR stays roughly constant while eyes are open, and drops sharply when closed.
If EAR stays below a threshold for too many consecutive frames -> alarm.

Run from the project root:
    python apps/11_drowsiness_detection.py
"""

import cv2
import sys
import os
from scipy.spatial import distance as dist

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "modules"))
from FaceMeshModule import FaceMeshDetector

# MediaPipe Face Mesh landmark indices around each eye (6 points each, same order
# as the classic EAR formula: [left corner, top1, top2, right corner, bottom2, bottom1])
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

EAR_THRESHOLD = 0.22        # below this = "eye closed"
CONSEC_FRAMES = 20          # ~0.6-0.8 sec at typical webcam fps


def eye_aspect_ratio(eye_points):
    """eye_points = list of 6 [x, y] coordinates in the fixed order above."""
    p1, p2, p3, p4, p5, p6 = eye_points
    vertical_1 = dist.euclidean(p2, p6)
    vertical_2 = dist.euclidean(p3, p5)
    horizontal = dist.euclidean(p1, p4)
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear


cam_width, cam_height = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, cam_width)
cap.set(4, cam_height)

detector = FaceMeshDetector(max_faces=1)

closed_frame_counter = 0
alarm_on = False

while True:
    success, img = cap.read()
    if not success:
        print("Could not read from webcam.")
        break

    img, faces = detector.find_face_mesh(img, draw=False)

    if faces:
        face = faces[0]
        # face is a list of [id, x, y] - build coordinate-only lookups for our eye points
        left_eye_pts = [(face[i][1], face[i][2]) for i in LEFT_EYE]
        right_eye_pts = [(face[i][1], face[i][2]) for i in RIGHT_EYE]

        left_ear = eye_aspect_ratio(left_eye_pts)
        right_ear = eye_aspect_ratio(right_eye_pts)
        avg_ear = (left_ear + right_ear) / 2.0

        # Draw eye outlines for visual feedback
        for pt in left_eye_pts + right_eye_pts:
            cv2.circle(img, pt, 2, (0, 255, 0), cv2.FILLED)

        cv2.putText(img, f"EAR: {avg_ear:.2f}", (10, 40),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)

        if avg_ear < EAR_THRESHOLD:
            closed_frame_counter += 1
            if closed_frame_counter >= CONSEC_FRAMES:
                alarm_on = True
        else:
            closed_frame_counter = 0
            alarm_on = False

        if alarm_on:
            cv2.putText(img, "DROWSINESS ALERT!", (100, 240),
                        cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 0, 255), 3)
            cv2.rectangle(img, (0, 0), (cam_width, cam_height), (0, 0, 255), 10)
            # Simple beep - works cross-platform without extra audio libraries.
            # On Windows this uses winsound; if not on Windows, this line will
            # just be skipped safely.
            try:
                import winsound
                winsound.Beep(1000, 200)
            except ImportError:
                pass
    else:
        closed_frame_counter = 0
        alarm_on = False

    cv2.imshow("Drowsiness Detection", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()