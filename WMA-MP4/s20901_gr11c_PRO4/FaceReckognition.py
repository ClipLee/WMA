import os
import pandas as pd
import cv2
import keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import Sequential
from keras.applications.densenet import layers
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

classes = ['jdepp', 'mietczynski', 'piszczek', 'pudzian']

input_shape = (150, 150, 3)

batch_size = 32
epochs = 10
data_augmentation = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
dataset_path = ''

def create_model():
    model = Sequential([
        layers.Rescaling(1 / 255, input_shape=(224, 224, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(3)
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # load the dataset
    dataset_path = 'train_set'

    model.summary()
    return model

model = create_model()
train = data_augmentation.flow_from_directory(
    os.path.join(dataset_path, 'train_set'),
    target_size=(input_shape[0], input_shape[1]),
    batch_size=batch_size,
    class_mode='categorical'
)

validation = ImageDataGenerator().flow_from_directory(
    os.path.join(dataset_path, 'test_set'),
    target_size=(input_shape[0], input_shape[1]),
    batch_size=batch_size,
    class_mode='categorical'
)
# Train the model
model.fit(
    train,
    steps_per_epoch=train.samples // batch_size,
    epochs=10,
    validation_data=validation,
    validation_steps=validation.samples // batch_size
)
model.save('trained_model.h5')

# It can be used to reconstruct the model identically.
reconstructed_model = keras.models.load_model("trained_model.h5")

# Let's check:
np.testing.assert_allclose(
    model.predict(test_input), reconstructed_model.predict(test_input)
)
# Load the model for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

# Load all images from the known_faces directory
known_faces = []
for filename in os.listdir('training_set'):
    image = cv2.imread(os.path.join('images', filename))
    known_faces.append(image)

# Load all images from the unknown_faces directory
for filename in os.listdir('training_set'):
    # Load the image we want to check
    unknown_image = cv2.imread(os.path.join('training_set', filename))

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(unknown_image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

    # array for name of the person in picture
    face_names = []

    # Loop through each face found in the unknown image
    for (x, y, w, h) in faces:
        # Crop the face from the image
        face_image = gray_image[y:y+h, x:x+w]

        # Resize the face to match the input size of the model
        resized_face_image = cv2.resize(face_image, (96, 96))

        # Normalize the pixel values of the resized face image
        normalized_face_image = resized_face_image / 255.0

        # Reshape  to match the input shape of the model
        reshaped_face_image = np.reshape(normalized_face_image, (1, 96, 96, 1))

        # Use the model to predict which person this face belongs to
        # Replace this with your own code for classification.
        name = "Unknown"

        # Add the name of the person in this picture to our list of names
        face_names.append(name)

    # Display the results
    for (x, y, w, h), name in zip(faces, face_names):
        # Draw a box around the face
        cv2.rectangle(unknown_image, (x, y), (x+w, y+h), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(unknown_image, (x-1, y-35), (x+w+1, y), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(unknown_image, name, (x+6, y-6), font, 1.0, (255, 255, 255), 1)

    # Display the final image with boxes drawn around each detected face and their names below them.
    cv2.imshow('Image', unknown_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
