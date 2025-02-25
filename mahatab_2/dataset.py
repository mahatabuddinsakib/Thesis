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

# Step 1: Preprocess Images
def preprocess_images(input_dir, output_dir, target_size=(224, 224)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, _, files in os.walk(input_dir):
        relative_path = os.path.relpath(root, input_dir)
        output_subdir = os.path.join(output_dir, relative_path)
        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)

        for img_name in files:
            try:
                img_path = os.path.join(root, img_name)
                img = tf_image.load_img(img_path, target_size=target_size)  # Load and resize
                img_resized = tf_image.img_to_array(img)  # Convert to array
                img_resized = np.expand_dims(img_resized, axis=0)  # Add batch dimension
                img_resized = tf.keras.applications.resnet50.preprocess_input(img_resized)  # Preprocess for ResNet
                output_path = os.path.join(output_subdir, img_name)
                tf.keras.preprocessing.image.save_img(output_path, img_resized[0])  # Save processed image
            except Exception as e:
                print(f"Error processing {img_name}: {e}")

    print(f"Processing complete. Processed images saved in: {output_dir}")
preprocess_images("cell", "processed_images", target_size=(224, 224))


