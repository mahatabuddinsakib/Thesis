import os
import numpy as np
import tensorflow as tf
from cv2 import transform
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image as tf_image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import torch
import matplotlib.pyplot as plt
import cv2


# Step 2: Dataset setup using TensorFlow
def generate_labels_from_dirs(base_dir):
    labels = []
    for label, folder in enumerate(['parasitized', 'uninfected']):
        folder_path = os.path.join(base_dir, folder)
        if not os.path.exists(folder_path):
            continue
        for img_name in os.listdir(folder_path):
            labels.append({'image': os.path.join(folder, img_name), 'label': label})
    return labels


# Step 3:Model build
# Load ResNet50 as feature extractor
base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_output = layers.GlobalAveragePooling2D()(base_model.output)

# First parallel CNN branch
cnn_branch_1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(base_model.output)
cnn_branch_1 = layers.MaxPooling2D((2, 2))(cnn_branch_1)
cnn_branch_1 = layers.Flatten()(cnn_branch_1)

# Second parallel CNN branch
cnn_branch_2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(base_model.output)
cnn_branch_2 = layers.MaxPooling2D((2, 2))(cnn_branch_2)
cnn_branch_2 = layers.Flatten()(cnn_branch_2)

# Attention Transformer Block
input_tensor = layers.Dense(256, activation='relu')(base_output)
query = layers.Dense(128, activation='relu')(input_tensor)
key = layers.Dense(128, activation='relu')(input_tensor)
value = layers.Dense(128, activation='relu')(input_tensor)
attention_output = layers.Attention()([query, key, value])
attention_output = layers.Flatten()(attention_output)

# Concatenation of branches and attention output
merged = layers.Concatenate()([cnn_branch_1, cnn_branch_2, attention_output])
final_output = layers.Dense(2, activation='softmax')(merged)

# Build and compile the model
model = models.Model(inputs=base_model.input, outputs=final_output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Summary
model.summary()

# Dataset
processed_image_dir = 'processed_images'
labels = generate_labels_from_dirs(processed_image_dir)
train_labels, val_labels = train_test_split(labels, test_size=0.2, random_state=42)

# Prepare data generators
train_datagen = ImageDataGenerator(rescale=1. / 255)
val_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(processed_image_dir, target_size=(224, 224), batch_size=32,
                                                    class_mode='categorical')
val_generator = val_datagen.flow_from_directory(processed_image_dir, target_size=(224, 224), batch_size=32,
                                                class_mode='categorical')

# Training the model
model.fit(train_generator, epochs=5, validation_data=val_generator)

# Save the trained model
model.save('malaria_model.h5')

