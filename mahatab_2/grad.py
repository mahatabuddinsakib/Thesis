import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as tf_image
import matplotlib.pyplot as plt
import cv2

# Load the trained model
model = tf.keras.models.load_model('malaria_model.h5')

# Choose the last convolutional layer
last_conv_layer = model.get_layer('conv5_block3_out')  # Adjust based on your model's architecture

# Create a new model that outputs both the last convolutional layer's activations and the model's predictions
grad_model = tf.keras.models.Model(inputs=model.input, outputs=[last_conv_layer.output, model.output])


# Define function to process and predict a single image
def predict_image(image_path):
    img = tf_image.load_img(image_path, target_size=(224, 224))  # Resize image to 224x224
    img_array = tf_image.img_to_array(img)  # Convert image to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Rescale image values to [0, 1]

    # Convert the image array to a TensorFlow tensor
    img_tensor = tf.convert_to_tensor(img_array)

    # Predict using the model
    prediction = model.predict(img_array)

    return img, prediction, img_tensor


# Grad-CAM visualization function
def generate_gradcam_heatmap(model, grad_model, img_tensor, predicted_class):
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        last_conv_layer_output, preds = grad_model(img_tensor)  # Get both activations and predictions
        class_channel = preds[:, predicted_class]

    grads = tape.gradient(class_channel,
                          last_conv_layer_output)  # Compute gradients with respect to the last conv layer
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # Global average pooling

    # Get the output of the last convolutional layer
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize the heatmap
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    return heatmap


# Superimpose heatmap on the image
def superimpose_heatmap_on_image(img, heatmap):
    img = np.array(img)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # Resize heatmap to image size
    heatmap = np.uint8(255 * heatmap)  # Convert to uint8 for visualization

    # Apply a colormap to the heatmap
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Superimpose the heatmap on the image
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    return superimposed_img


sample_folder = 'sample'

# Predict and visualize Grad-CAM on all images in the 'sample' folder
for img_name in os.listdir(sample_folder):
    img_path = os.path.join(sample_folder, img_name)
    if os.path.isfile(img_path):
        img, prediction, img_tensor = predict_image(img_path)

        # Get predicted class
        predicted_class = np.argmax(prediction)
        class_labels = ['parasitized', 'uninfected']
        print(f"Image: {img_name} | Predicted Class: {class_labels[predicted_class]}")

        # Generate Grad-CAM heatmap
        heatmap = generate_gradcam_heatmap(model, grad_model, img_tensor, predicted_class)

        # Superimpose heatmap on image
        superimposed_img = superimpose_heatmap_on_image(img, heatmap)

        # Display the original image and Grad-CAM visualization
        plt.figure(figsize=(10, 5))

        # Original image
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title(f"Predicted: {class_labels[predicted_class]}")

        # Grad-CAM visualization
        plt.subplot(1, 2, 2)
        plt.imshow(superimposed_img)
        plt.title(f"Grad-CAM: {class_labels[predicted_class]}")

        plt.show()
