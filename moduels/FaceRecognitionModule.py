"""
FaceRecognitionModule.py
--------------------------
Wraps the face_recognition library.
Loads known faces from data/known_faces/ (one clear photo per person,
filename = their name, e.g. "John_Smith.jpg") and can identify faces
in a live webcam frame.

Used for: Face Attendance System.
"""

import face_recognition
import os
import cv2
import numpy as np


class FaceRecognizer:
    def __init__(self, known_faces_dir):
        self.known_encodings = []
        self.known_names = []
        self._load_known_faces(known_faces_dir)

    def _load_known_faces(self, folder):
        if not os.path.exists(folder):
            print(f"Warning: {folder} does not exist. No known faces loaded.")
            return

        for filename in os.listdir(folder):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            path = os.path.join(folder, filename)
            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)

            if len(encodings) == 0:
                print(f"Warning: no face found in {filename}, skipping.")
                continue

            name = os.path.splitext(filename)[0].replace("_", " ")
            self.known_encodings.append(encodings[0])
            self.known_names.append(name)

        print(f"Loaded {len(self.known_names)} known face(s): {self.known_names}")

    def identify_faces(self, frame, resize_scale=0.25, tolerance=0.5):
        """
        Detects and identifies faces in a frame.
        Returns a list of (name, (top, right, bottom, left)) in ORIGINAL frame coordinates.
        Frame is downscaled internally for speed, then coordinates are scaled back up.
        """
        small_frame = cv2.resize(frame, (0, 0), fx=resize_scale, fy=resize_scale)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small)
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

        results = []
        for encoding, location in zip(face_encodings, face_locations):
            name = "Unknown"
            if self.known_encodings:
                distances = face_recognition.face_distance(self.known_encodings, encoding)
                best_match = np.argmin(distances)
                if distances[best_match] < tolerance:
                    name = self.known_names[best_match]

            # Scale face location back up to original frame size
            top, right, bottom, left = location
            scale = int(1 / resize_scale)
            top, right, bottom, left = top * scale, right * scale, bottom * scale, left * scale
            results.append((name, (top, right, bottom, left)))

        return results