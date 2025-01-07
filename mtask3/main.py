import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import pytesseract

# Load the EMNIST dataset (or MNIST for simplicity)
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the images to be between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# Reshape images to match the input shape for the CNN
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

# Create a simple CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # Output layer for 10 classes (digits 0-9)
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))

# Save the model
model.save('handwritten_character_model.h5')

# Function to predict characters from an image
def predict_handwritten_characters(image_path):
    # Read and preprocess the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (28, 28))  # Resize to match the input shape of the model
    image = image / 255.0  # Normalize the image
    image = image.reshape(1, 28, 28, 1)  # Reshape for the model

    # Predict the character
    prediction = model.predict(image)
    predicted_label = np.argmax(prediction)

    return predicted_label

# Test the model with an image
image_path = r'C:\local_d_data\6thsmester\codeAlpha\mtask3\WhatsApp Image 2024-11-23 at 20.21.50_a22eb5e3.jpg'
predicted_label = predict_handwritten_characters(image_path)
print("Predicted Handwritten Character: ", predicted_label)

# Now integrate Tesseract OCR to extract text from the image
# Set the correct path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Read the image
image = cv2.imread(image_path)

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply binarization or thresholding
_, binarized_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY)

# Apply OCR to detect text
extracted_text = pytesseract.image_to_string(binarized_image, config='--psm 6')

# Output the detected text
print("Detected Handwritten Text (OCR): ", extracted_text)
