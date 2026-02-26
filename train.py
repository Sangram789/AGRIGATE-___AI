import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU (remove if using GPU)

import numpy as np
import cv2
import pickle
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation
from tensorflow.keras.layers import Flatten, Dropout, Dense, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

# ======================
# CONFIG
# ======================
EPOCHS = 30
INIT_LR = 1e-3
BS = 16
IMAGE_SIZE = (128, 128)

# ✅ FIXED PATH (Use forward slash or raw string)
dataset_path = "E:\CropDisease\dataset\Potato Leaf Disease"

# ======================
# LOAD IMAGES
# ======================
print("[INFO] Loading images...")

image_list = []
label_list = []

def load_image(path):
    image = cv2.imread(path)
    if image is None:
        return None
    image = cv2.resize(image, IMAGE_SIZE)
    return image

for folder in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder)

    if not os.path.isdir(folder_path):
        continue

    print(f"[INFO] Processing {folder}")

    for file in os.listdir(folder_path):
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(folder_path, file)
            img = load_image(img_path)
            if img is not None:
                image_list.append(img)
                label_list.append(folder)

print("[INFO] Image loading completed")

# ======================
# PREPARE DATA
# ======================
X = np.array(image_list, dtype="float32") / 255.0

lb = LabelBinarizer()
y = lb.fit_transform(label_list)

pickle.dump(lb, open("label_transform.pkl", "wb"))

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ======================
# DATA AUGMENTATION
# ======================
aug = ImageDataGenerator(
    rotation_range=25,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

# ======================
# BUILD MODEL
# ======================
print("[INFO] Building model...")

model = Sequential()

model.add(Conv2D(32, (3,3), padding="same", input_shape=(128,128,3)))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3,3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3,3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))

model.add(Dense(len(lb.classes_), activation="softmax"))

# ======================
# COMPILE MODEL
# ======================
opt = Adam(learning_rate=INIT_LR)

model.compile(
    loss="categorical_crossentropy",
    optimizer=opt,
    metrics=["accuracy"]
)

model.summary()

# ======================
# CALLBACK
# ======================
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=0.00001
)

# ======================
# TRAIN MODEL
# ======================
print("[INFO] Training model...")

history = model.fit(
    aug.flow(x_train, y_train, batch_size=BS),
    validation_data=(x_test, y_test),
    steps_per_epoch=max(1, len(x_train) // BS),  # prevents division issue
    epochs=EPOCHS,
    callbacks=[reduce_lr],
    verbose=1
)

# ======================
# SAVE MODEL
# ======================
print("[INFO] Saving model...")
model.save("crop_disease_model.h5")

# ======================
# EVALUATE
# ======================
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Accuracy: {acc*100:.2f}%")