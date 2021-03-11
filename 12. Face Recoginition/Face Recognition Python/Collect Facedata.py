# First Collecting Data about different Faces

import cv2
import numpy as np 
import os

cap = cv2.VideoCapture(0)
skip = 0
face_data = list()

dataPath = './Faces Datasets/' # to store faces in the form of np array
file_name = input('Enter name of face: ')
# Loading Haarcascade classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

# Sort images on the basis of width(w) and height(h)
# Select largest area face
# Frame is an image
# Frames have coordinates like [Y-axis, X-axis]
# 'offset = 10' below is like padding of 10px around a face

while True:
    ret, frame = cap.read()

    # If there is something wrong with camera then continue
    if ret is False:
        continue
# scaleFactor=1.3, minNeighbors=4
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors = 5)

    for face in faces:
        x,y,w,h = face
        # cv2.rectangle is to draw rectangle around detected face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),2)
        # Extract(Crop face from that rectangle known as 
        # Region Of Interest ROI)
        offset = 10 #offset is like padding of 10

        face_section = frame[y-offset:y+h+offset, x - offset:x+w+offset]
        skip += 1
        if skip%10 == 0: # store every 10th face to save space
            face_data.append(face_section)
            print(len(face_data))

    cv2.imshow('Frame', frame)
    cv2.imshow('Face Section', face_section)
#     #Store every 10th face
#     if (skip%10==0):
#         #store the 10th face later on
#         pass
    
    
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed is ord('q'):
        break

# Convert our facelist array into a numpy array

face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0], -1))

print(face_data.shape)

# Save this data to file system
np.save(dataPath+file_name+'.npy', face_data)
print('Data successfully saved at:'+ dataPath+file_name+'.npy')
cap.release()
cv2.destoryAllWindows()
