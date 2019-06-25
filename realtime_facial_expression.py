#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 15:36:49 2017

@author: adam
"""

import cv2
import numpy as np
from keras.models import load_model

##Satart Section
''' Keras took all GPU memory so to limit GPU usage, I have add those lines'''

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()  # Create a tensorflow configurator
config.gpu_options.per_process_gpu_memory_fraction = 0.1  # Configure Tensorflow to use only 10% of GPU memory per process
set_session(tf.Session(config=config))  # Start the Tensorflow session
''' Keras took all GPU memory so to limit GPU usage, I have add those lines'''
## End section

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml') #  Information for detecting face edges

video_capture = cv2.VideoCapture(0)  # Start capturing video from the webcam
model = load_model('keras_model/model_5-49-0.62.hdf5') #  training files containing faces and their corresponding
model.get_config()

# Names of the emotions encoded as number. For example, emotion 0 is angry and 1 is disgust:
target = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
font = cv2.FONT_HERSHEY_SIMPLEX  # Create an object represeting the font Hershey Simplex
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()  # Capture a video frame from the webcam into an array

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert the image array to gray scale

    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1)  # Detect the faces in the shapshot

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        # Iterates over the edges of each face, in each case storing x_start,
        # y_start, width and height as x, y, w, and h respectively
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2, 5)  # Draws the rectangle around the face
        face_crop = frame[y:y + h, x:x + w]  # Crops the image to the face and its enclosing rectangle
        face_crop = cv2.resize(face_crop, (48, 48))  # resizes the cropped image to a 48*48 px image
        face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY) # convert the image array to gray scale
        face_crop = face_crop.astype('float32') / 255  # converts the image array to floats from 0 to 1
        face_crop = np.asarray(face_crop)  # Converts the image array to a numpy array
        face_crop = face_crop.reshape(1, 1, face_crop.shape[0], face_crop.shape[1])  #Adds two extra dimensions, with length 1, to the start of the dimensionality of the array, ending up in (1, 1, 48, 48
        result = target[np.argmax(model.predict(face_crop))]  # We predict the likelinesses of each emotion, and select the most likely emotion
        cv2.putText(frame, result, (x, y), font, 1, (200, 0, 0), 3, cv2.LINE_AA)  # Put a label saying the emotion name onto the image

    # Display the resulting frame
    cv2.imshow('Video', frame) #Show in the screen the captured image pus the emotion label

    if cv2.waitKey(1) & 0xFF == ord('q'):
        #If the user press Ctrl-Q, stop the video capturing
        break

# When everything is done, release the capture
video_capture.release() #Stop capturing video
cv2.destroyAllWindows() #  Close al the video-capturing windows
