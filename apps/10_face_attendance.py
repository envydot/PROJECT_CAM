"""
10_face_attendance.py
-------------------------
Recognizes known faces from the webcam and logs their name + timestamp
to data/attendance.csv - once per person per day (no duplicate spam entries).

SETUP BEFORE RUNNING:
    1. Put one clear, front-facing photo per person in data/known_faces/
       Filename = their name, e.g.:
           data/known_faces/John_Smith.jpg
           data/known_faces/Priya_Patel.jpg
    2. Good lighting + face clearly visible in the photo = better accuracy.

Run from the project root:
    python apps/10_face_attendance.py
"""

import cv2
import sys
import os
import csv
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "modules"))
from FaceRecognitionModule import FaceRecognizer

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
KNOWN_FACES_DIR = os.path.join(BASE_DIR, "data", "known_faces")
ATTENDANCE_FILE = os.path.join(BASE_DIR, "data", "attendance.csv")

os.makedirs(os.path.dirname(ATTENDANCE_FILE), exist_ok=True)
if not os.path.exists(ATTENDANCE_FILE):
    with open(ATTENDANCE_FILE, "w", newline="") as f:
        csv.writer(f).writerow(["Name", "Date", "Time"])


def already_marked_today(name):
    today = datetime.now().strftime("%Y-%m-%d")
    with open(ATTENDANCE_FILE, "r") as f:
        for row in csv.reader(f):
            if len(row) >= 2 and row[0] == name and row[1] == today:
                return True
    return False


def mark_attendance(name):
    now = datetime.now()
    with open(ATTENDANCE_FILE, "a", newline="") as f:
        csv.writer(f).writerow([name, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")])
    print(f"Marked attendance: {name} at {now.strftime('%H:%M:%S')}")


recognizer = FaceRecognizer(KNOWN_FACES_DIR)

cap = cv2.VideoCapture(0)
cap.set(3, 960)
cap.set(4, 540)

marked_this_session = set()  # avoid re-checking the CSV every single frame

while True:
    success, img = cap.read()
    if not success:
        print("Could not read from webcam.")
        break

    results = recognizer.identify_faces(img)

    for name, (top, right, bottom, left) in results:
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(img, (left, top), (right, bottom), color, 2)
        cv2.rectangle(img, (left, bottom - 25), (right, bottom), color, cv2.FILLED)
        cv2.putText(img, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

        if name != "Unknown" and name not in marked_this_session:
            if not already_marked_today(name):
                mark_attendance(name)
            marked_this_session.add(name)

    cv2.putText(img, "q = quit", (10, 25), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)
    cv2.imshow("Face Attendance", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()