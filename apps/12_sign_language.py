"""
12_sign_language.py
-----------------------
All-in-one sign language tool: collect data, train the model, and
recognize signs live - all from one file with a simple menu.

Run from the project root:
    python apps/12_sign_language.py

You'll still need to do the 3 steps in order the FIRST time:
    1) Collect data for each sign
    2) Train the model
    3) Recognize live
But now it's all one file - just pick the option each time.
"""

import cv2
import sys
import os
import csv
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "modules"))
from HandTrackingModule import HandDetector

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_FILE = os.path.join(BASE_DIR, "data", "sign_language_data.csv")
MODEL_FILE = os.path.join(BASE_DIR, "data", "sign_model.pkl")
os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)


def normalize_landmarks(landmark_list):
    """Coordinates relative to the wrist, so hand position in frame doesn't matter."""
    wrist_x, wrist_y = landmark_list[0][1], landmark_list[0][2]
    flat = []
    for _, x, y in landmark_list:
        flat.append(x - wrist_x)
        flat.append(y - wrist_y)
    return flat


# ============================================================
# MODE 1: COLLECT DATA
# ============================================================
def collect_data():
    if not os.path.exists(DATA_FILE):
        with open(DATA_FILE, "w", newline="") as f:
            header = ["label"] + [f"{axis}{i}" for i in range(21) for axis in ("x", "y")]
            csv.writer(f).writerow(header)

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    detector = HandDetector(max_hands=1, detection_con=0.8)

    current_label = input("Enter the label for the first sign (e.g. A, Hello, Yes): ").strip()
    sample_count = 0
    print("SPACE = capture a sample | n = new label | q = done collecting")

    while True:
        success, img = cap.read()
        if not success:
            break
        img = cv2.flip(img, 1)
        img = detector.find_hands(img)
        landmark_list = detector.find_position(img, draw=False)

        cv2.putText(img, f"Label: {current_label}  Samples: {sample_count}", (10, 30),
                    cv2.FONT_HERSHEY_PLAIN, 1.8, (0, 255, 0), 2)
        cv2.putText(img, "SPACE=capture  n=new label  q=done", (10, 460),
                    cv2.FONT_HERSHEY_PLAIN, 1.3, (255, 255, 255), 1)
        cv2.imshow("Collect Sign Data", img)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("n"):
            current_label = input("Enter the label for the next sign: ").strip()
            sample_count = 0
        elif key == ord(" ") and len(landmark_list) != 0:
            row = [current_label] + normalize_landmarks(landmark_list)
            with open(DATA_FILE, "a", newline="") as f:
                csv.writer(f).writerow(row)
            sample_count += 1
            print(f"Captured sample #{sample_count} for '{current_label}'")

    cap.release()
    cv2.destroyAllWindows()
    print(f"Data saved to {DATA_FILE}")


# ============================================================
# MODE 2: TRAIN MODEL
# ============================================================
def train_model():
    if not os.path.exists(DATA_FILE):
        print(f"No data found at {DATA_FILE}. Run option 1 (Collect Data) first.")
        return

    df = pd.read_csv(DATA_FILE)
    X = df.drop("label", axis=1)
    y = df["label"]
    print(f"Loaded {len(df)} samples across {y.nunique()} sign(s): {sorted(y.unique())}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print(f"\nTest accuracy: {accuracy_score(y_test, predictions):.2%}\n")
    print(classification_report(y_test, predictions))

    joblib.dump(model, MODEL_FILE)
    print(f"Model saved to {MODEL_FILE}")


# ============================================================
# MODE 3: RECOGNIZE LIVE
# ============================================================
def recognize_live():
    if not os.path.exists(MODEL_FILE):
        print(f"No trained model found at {MODEL_FILE}. Run option 2 (Train Model) first.")
        return

    model = joblib.load(MODEL_FILE)
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    detector = HandDetector(max_hands=1, detection_con=0.8)

    while True:
        success, img = cap.read()
        if not success:
            break
        img = cv2.flip(img, 1)
        img = detector.find_hands(img)
        landmark_list = detector.find_position(img, draw=False)

        if len(landmark_list) != 0:
            features = normalize_landmarks(landmark_list)
            prediction = model.predict([features])[0]
            confidence = max(model.predict_proba([features])[0]) * 100

            cv2.rectangle(img, (20, 20), (320, 90), (0, 0, 0), cv2.FILLED)
            cv2.putText(img, f"{prediction} ({confidence:.0f}%)", (35, 70),
                        cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

        cv2.putText(img, "q = quit", (10, 460), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)
        cv2.imshow("Sign Language Recognition", img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ============================================================
# MENU
# ============================================================
if __name__ == "__main__":
    print("\n=== Sign Language Tool ===")
    print("1) Collect data for a new sign")
    print("2) Train the model")
    print("3) Recognize signs live")
    choice = input("Choose an option (1/2/3): ").strip()

    if choice == "1":
        collect_data()
    elif choice == "2":
        train_model()
    elif choice == "3":
        recognize_live()
    else:
        print("Invalid choice. Run the script again and pick 1, 2, or 3.")