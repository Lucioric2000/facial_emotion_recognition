# -*- coding: utf-8 -*-

import cv2
import numpy as np
from keras.models import load_model
import sys

##Satart Section
''' Keras took all GPU memory so to limit GPU usage, I have add those lines'''

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()  # Create a tensorflow configurator
config.gpu_options.per_process_gpu_memory_fraction = 0.1 #Configure Tensorflow to use only 10% of GPU memory per process
set_session(tf.Session(config=config))  # Start the Tensorflow session
''' Keras took all GPU memory so to limit GPU usage, I have add those lines'''
## End section


faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml') #  Information for detecting face edges
model = load_model('keras_model/model_5-49-0.62.hdf5') #  training files containing faces and their corresponding
# emotion, among angry, disgust, fear, happy, sad, surprise and neutral

def test_image(addr):
    # Names of the emotions encoded as number. For example, emotion 0 is angry and 1 is disgust:
    target = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    font = cv2.FONT_HERSHEY_SIMPLEX  # Create an object represeting the font Hershey Simplex

    im = cv2.imread(addr)  # read the image from the file and store in an array
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # convert the image array to gray scale
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1)  # Detect the faces in an image

    for (x, y, w, h) in faces:
            # Iterates over the edges of each face, in each case storing x_start,
            # y_start, width and height as x, y, w, and h respectively
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2, 5)  # Draws the rectangle around the face
            face_crop = im[y:y + h, x:x + w]  # Crops the image to the face and its enclosing rectangle
            face_crop = cv2.resize(face_crop, (48, 48))  # resizes the cropped image to a 48*48 px image
            face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)  # convert the image array to gray scale
            face_crop = face_crop.astype('float32') / 255  # converts the image array to floats from 0 to 1
            face_crop = np.asarray(face_crop)  # Converts the image array to a numpy array
            face_crop = face_crop.reshape(1, 1, face_crop.shape[0], face_crop.shape[1])  # Adds two extra dimensions, with length 1, to the start of the dimensionality of the array, ending up in (1, 1, 48, 48)
            result = target[np.argmax(model.predict(face_crop))]  # We predict the likelinesses of each emotion, and select the most likely emotion
            cv2.putText(im, result, (x, y), font, 1, (200, 0, 0), 3, cv2.LINE_AA)  # Put a label saying the emotion name onto the image
    cv2.imshow('result', im)  # Show the image with the emotion label
    cv2.imwrite('result_emotion_detection_app.jpg', im)  # Save the image with the emotion label
    cv2.waitKey(0)  # Wait until a key is pressed


if __name__=='__main__':
    if len(sys.argv)>1:
        image_address = sys.argv[1]  # get the image file name from the command line first argument
    else:
        image_address = "tes.jpg"
    test_image(image_address)  # Test the image to see which kind of expression do it contain
