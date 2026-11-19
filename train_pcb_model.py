import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# 1. CONFIGURATION
DATASET_PATH = os.path.join("PCB_DATASET", "images")
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 25

# Defect classes matching the PKU PCB Dataset & your abstract
CLASSES = ["Missing_hole", "Mouse_bite", "Open_circuit", "Short", "Spur", "Spurious_copper"]

print("Loading and preprocessing PCB images...")

images = []
labels = []

# 2. IMAGE PREPROCESSING (Noise Removal, Resizing, Normalization)
for class_idx, class_name in enumerate(CLASSES):
    class_folder = os.path.join(DATASET_PATH, class_name)
    if not os.path.exists(class_folder):
        raise FileNotFoundError(f"Folder not found: {class_folder}. Check your PCB_DATASET path.")
        
    for img_file in os.listdir(class_folder):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(class_folder, img_file)
            
            # Read image
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Noise Removal using Gaussian Blur (as specified in abstract)
            img = cv2.GaussianBlur(img, (3, 3), 0)
            
            # Resize to uniform dimensions
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            
            # Normalization: Scale pixel values to [0, 1]
            img = img.astype("float32") / 255.0
            
            images.append(img)
            labels.append(class_idx)

X = np.array(images)
y = tf.keras.utils.to_categorical(np.array(labels), num_classes=len(CLASSES))

print(f"Total processed images: {len(X)}")

# Split into 80% Training and 20% Testing/Validation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=labels
)

# 3. DATA AUGMENTATION
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.1
)

# 4. CONVOLUTIONAL NEURAL NETWORK (CNN) ARCHITECTURE
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(CLASSES), activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# 5. TRAIN THE MODEL
checkpoint = ModelCheckpoint("pcb_defect_cnn.h5", monitor="val_accuracy", save_best_only=True, verbose=1)
early_stop = EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)

print("Starting CNN training...")
history = model.fit(
    train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop]
)

# 6. EXPERIMENTAL EVALUATION & REPORT GENERATION
print("\nEvaluating model on unseen test images...")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=CLASSES))

# Save Accuracy & Loss Graphs for your final project report
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('CNN Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('CNN Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig("training_performance.png")
print("Saved training graphs to 'training_performance.png' and best model to 'pcb_defect_cnn.h5'.")=
