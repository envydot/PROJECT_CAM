"""
05_rock_paper_scissors.py
---------------------------
Play Rock Paper Scissors against the computer using hand gestures.

Gestures:
    Rock     = fist (all fingers down)
    Paper    = all 5 fingers up (open palm)
    Scissors = index + middle up, others down

Controls:
    SPACE = lock in your move / play a round
    q     = quit

Run from the project root:
    python apps/05_rock_paper_scissors.py
"""

import cv2
import sys
import os
import random
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "modules"))
from HandTrackingModule import HandDetector

# --- Setup ---
cam_width, cam_height = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, cam_width)
cap.set(4, cam_height)

detector = HandDetector(max_hands=1, detection_con=0.8)

choices = ["Rock", "Paper", "Scissors"]
player_score = 0
computer_score = 0
result_text = ""
result_timer = 0  # frames left to keep showing the result


def classify_gesture(fingers):
    """Turns a [thumb, index, middle, ring, pinky] list into Rock/Paper/Scissors/None."""
    total_up = fingers.count(1)

    if total_up == 0:
        return "Rock"
    elif total_up == 5:
        return "Paper"
    elif fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0:
        return "Scissors"
    else:
        return None  # not a recognized gesture


def decide_winner(player, computer):
    if player == computer:
        return "Tie"
    wins_against = {"Rock": "Scissors", "Paper": "Rock", "Scissors": "Paper"}
    if wins_against[player] == computer:
        return "Player"
    return "Computer"


while True:
    success, img = cap.read()
    if not success:
        print("Could not read from webcam.")
        break

    img = detector.find_hands(img)
    landmark_list = detector.find_position(img, draw=False)

    current_gesture = None
    if len(landmark_list) != 0:
        fingers = detector.fingers_up()
        current_gesture = classify_gesture(fingers)

    # --- Scoreboard ---
    cv2.rectangle(img, (0, 0), (cam_width, 50), (0, 0, 0), cv2.FILLED)
    cv2.putText(img, f"You: {player_score}   Computer: {computer_score}", (10, 35),
                cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

    # --- Show detected gesture live ---
    if current_gesture:
        cv2.putText(img, f"Detected: {current_gesture}", (10, cam_height - 20),
                     cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    else:
        cv2.putText(img, "Show Rock, Paper, or Scissors", (10, cam_height - 20),
                     cv2.FONT_HERSHEY_PLAIN, 2, (0, 165, 255), 2)

    # --- Show last round result for a couple seconds ---
    if result_timer > 0:
        cv2.putText(img, result_text, (10, 90), cv2.FONT_HERSHEY_PLAIN, 2.5, (0, 255, 255), 3)
        result_timer -= 1

    cv2.putText(img, "SPACE = play   q = quit", (10, cam_height - 60),
                cv2.FONT_HERSHEY_PLAIN, 1.5, (200, 200, 200), 1)

    cv2.imshow("Rock Paper Scissors", img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

    if key == ord(" ") and current_gesture:
        computer_choice = random.choice(choices)
        winner = decide_winner(current_gesture, computer_choice)

        if winner == "Player":
            player_score += 1
            result_text = f"You: {current_gesture} vs CPU: {computer_choice} -> You win!"
        elif winner == "Computer":
            computer_score += 1
            result_text = f"You: {current_gesture} vs CPU: {computer_choice} -> CPU wins!"
        else:
            result_text = f"You: {current_gesture} vs CPU: {computer_choice} -> Tie!"

        result_timer = 90  # show result for ~90 frames (~3 seconds at 30fps)

cap.release()
cv2.destroyAllWindows()