"""
HandTrackingModule.py
----------------------
A reusable wrapper around MediaPipe Hands.
This ONE file powers: finger counter, virtual mouse, volume control,
brightness control, RPS game, drawing board, gesture presentation,
gesture media player, and sign language recognition.

You will not need to touch this file again once it works.
"""

import cv2
import mediapipe as mp
import math


class HandDetector:
    def __init__(self, mode=False, max_hands=2, detection_con=0.7, track_con=0.7):
        """
        mode: False = treats video as a stream (faster, tracks between frames)
        max_hands: how many hands to detect at once
        detection_con: minimum confidence to detect a hand
        track_con: minimum confidence to keep tracking a hand
        """
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=mode,
            max_num_hands=max_hands,
            min_detection_confidence=detection_con,
            min_tracking_confidence=track_con,
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Landmark IDs for the tip of each finger (MediaPipe's numbering)
        self.tip_ids = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky
        self.results = None
        self.landmark_list = []

    def find_hands(self, img, draw=True):
        """Detects hands in the frame and optionally draws the skeleton on it."""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(
                        img, hand_lms, self.mp_hands.HAND_CONNECTIONS
                    )
        return img

    def find_position(self, img, hand_no=0, draw=True):
        """
        Returns a list of [id, x, y] for all 21 landmarks of the chosen hand.
        id 0 = wrist, 4 = thumb tip, 8 = index tip, etc.
        """
        self.landmark_list = []

        if self.results and self.results.multi_hand_landmarks:
            if hand_no < len(self.results.multi_hand_landmarks):
                hand = self.results.multi_hand_landmarks[hand_no]
                h, w, c = img.shape

                for idx, lm in enumerate(hand.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    self.landmark_list.append([idx, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        return self.landmark_list

    def fingers_up(self):
        """
        Returns a list of 5 values (0 or 1) representing whether each finger
        is up: [thumb, index, middle, ring, pinky].
        Must call find_position() first in the same frame.
        """
        fingers = []
        if not self.landmark_list:
            return [0, 0, 0, 0, 0]

        # Thumb: compare x-coordinates (thumb moves sideways, not up/down)
        if self.landmark_list[self.tip_ids[0]][1] > self.landmark_list[self.tip_ids[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Other 4 fingers: compare y-coordinates (tip above the joint below it = up)
        for id in range(1, 5):
            if self.landmark_list[self.tip_ids[id]][2] < self.landmark_list[self.tip_ids[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def find_distance(self, p1, p2, img, draw=True):
        """
        Distance between two landmark points (e.g. thumb tip=4, index tip=8).
        Used later for pinch-click, volume control, brightness control.
        """
        x1, y1 = self.landmark_list[p1][1], self.landmark_list[p1][2]
        x2, y2 = self.landmark_list[p2][1], self.landmark_list[p2][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.circle(img, (cx, cy), 8, (0, 255, 0), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        return length, img, [x1, y1, x2, y2, cx, cy]