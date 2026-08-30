"""
09a_train_mask_model.py
--------------------------
Trains a mask/no-mask classifier using transfer learning on MobileNetV2.
Run this ONCE to produce data/mask_model.h5, which 09_mask_detection.py then uses live.

BEFORE RUNNING, download a mask dataset (free, e.g. search "Face Mask Detection
Dataset" on Kaggle) and arrange it like this:

    data/mask_dataset/
        with_mask/       <- images of people wearing masks
        without_mask/     <- images of people not wearing masks

This is a one-time training step - it can take 10-30+ minutes depending on
your machine and dataset size. You only need to run it once.

Run from the project root:
    python apps/09a_train_mask_model.py
"""

import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATASET_DIR = os.path.join(BASE_DIR, "data", "mask_dataset")
MODEL_OUT = os.path.join(BASE_DIR, "data", "mask_model.h5")

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10

if not os.path.exists(DATASET_DIR):
    raise FileNotFoundError(
        f"Dataset not found at {DATASET_DIR}. "
        "Download a mask/no-mask image dataset and put it there with "
        "with_mask/ and without_mask/ subfolders before running this script."
    )

# --- Data loading with augmentation (helps the model generalize from limited images) ---
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=15,
    zoom_range=0.15,
    horizontal_flip=True,
    validation_split=0.2,
)

train_gen = datagen.flow_from_directory(
    DATASET_DIR, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
    class_mode="categorical", subset="training",
)
val_gen = datagen.flow_from_directory(
    DATASET_DIR, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
    class_mode="categorical", subset="validation",
)

print("Class mapping:", train_gen.class_indices)  # e.g. {'with_mask': 0, 'without_mask': 1}

# --- Build model: pretrained MobileNetV2 base + small custom head ---
base_model = MobileNetV2(weights="imagenet", include_top=False,
                          input_tensor=Input(shape=(IMG_SIZE, IMG_SIZE, 3)))
base_model.trainable = False  # freeze pretrained layers, only train our new head

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
predictions = Dense(2, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(learning_rate=1e-4), loss="categorical_crossentropy",
              metrics=["accuracy"])

model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
model.save(MODEL_OUT)
print(f"Model saved to {MODEL_OUT}")