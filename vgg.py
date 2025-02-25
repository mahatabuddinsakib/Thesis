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
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import VGG19


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
# Load VGG19 as feature extractor
base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base model
# Extract features
x = layers.GlobalAveragePooling2D()(base_model.output)
# First CNN branch
cnn_branch_1 = layers.Dense(512, activation='relu')(x)
cnn_branch_1 = layers.Dense(256, activation='relu')(cnn_branch_1)

# Second CNN branch
cnn_branch_2 = layers.Dense(512, activation='relu')(x)
cnn_branch_2 = layers.Dense(256, activation='relu')(cnn_branch_2)

# Concatenate CNN branches
concat_cnn = layers.Concatenate()([cnn_branch_1, cnn_branch_2])
# Reshape for Transformer input
transformer_input = layers.Reshape((1, 512))(concat_cnn)

# Multi-Head Attention Transformer
attention = layers.MultiHeadAttention(num_heads=4, key_dim=128)(transformer_input, transformer_input)
attention = layers.GlobalAveragePooling1D()(attention)

# Final dense layers
x = layers.Dense(128, activation='relu')(attention)
x = layers.Dropout(0.5)(x)
out = layers.Dense(2, activation='softmax')(x)  # 2 classes  # 2 classes: infected (1) and uninfected (0)

model = models.Model(inputs=base_model.input, outputs=x)

# Move the model to the GPU if available
device = 'cuda' if tf.config.list_physical_devices('GPU') else 'cpu'
print(f"Using device: {device}")
model = model if device == 'cuda' else model  # TensorFlow automatically uses GPU if available

# Loss and optimizer
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Print model summary
model.summary()
# Dataset
processed_image_dir = 'processed_images'
labels = generate_labels_from_dirs(processed_image_dir)
train_labels, val_labels = train_test_split(labels, test_size=0.2, random_state=42)

# Prepare data generators
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(processed_image_dir,target_size=(224, 224),batch_size=32,class_mode="sparse")  # Ensures labels are one-hot encoded
val_generator = val_datagen.flow_from_directory(processed_image_dir,target_size=(224, 224),batch_size=32,class_mode="sparse")

# Training the model
model.fit(train_generator, epochs=5, validation_data=val_generator)

# Save the trained model
model.save('malaria_model.h5')

