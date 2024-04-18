import os

import cv2
import numpy as np
from keras_vggface.vggface import VGGFace


def main():
    # Pretrained model
    model = VGGFace(model='resnet50')

    face_cascade = cv2.CascadeClassifier('/haarcascade_frontalface_default.xml')

    # Loop through each img
    for filename in os.listdir('../s20901_gr11c_PRO4/images'):

        img = cv2.imread(os.path.join('../s20901_gr11c_PRO4/images', filename))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        # Loop through each face and classify it
        for (x, y, w, h) in faces:
            face = img[y:y+h, x:x+w]

            face = cv2.resize(face, (224, 224))

            face = np.expand_dims(face, axis=0)
            face = face.astype('float32')
            face /= 255.0
            prediction = model.predict(face)
            print("File: ", filename)
            print("Class: ", np.argmax(prediction))
            print("Confidence: ", np.max(prediction))

if __name__ == '__main__':
    main()
