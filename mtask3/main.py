import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import pytesseract
import os
# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize images
train_images = train_images / 255.0
test_images = test_images / 255.0

# Reshape images
train_images = train_images.reshape((-1, 28, 28, 1))
test_images = test_images.reshape((-1, 28, 28, 1))

# Define model path
MODEL_PATH = 'handwritten_character_model.h5'

# Check if model exists to avoid retraining
if os.path.exists(MODEL_PATH):
    model = models.load_model(MODEL_PATH)
    print("Model loaded from disk.")
else:
    # Create and train model
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))
    model.save(MODEL_PATH)
    print("Model trained and saved.")

# Function to predict handwritten character
def predict_handwritten_character(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Error loading image. Ensure the path is correct.")
    
    image = cv2.resize(image, (28, 28))
    image = image / 255.0  # Normalize
    image = image.reshape(1, 28, 28, 1)
    prediction = model.predict(image)
    return np.argmax(prediction)

# Path to test image
image_path = r'mtask3\WhatsApp Image 2024-11-23 at 20.21.50_a22eb5e3.jpg'

try:
    predicted_label = predict_handwritten_character(image_path)
    print("Predicted Handwritten Character:", predicted_label)
except Exception as e:
    print("Error:", e)

# Configure Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# OCR Processing
try:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Error loading image for OCR.")
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binarized_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    extracted_text = pytesseract.image_to_string(binarized_image, config='--psm 6')
    print("Detected Handwritten Text (OCR):", extracted_text.strip())
except Exception as e:
    print("OCR Error:", e)
