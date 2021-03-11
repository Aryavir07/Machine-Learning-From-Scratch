import cv2
import numpy as np

# Simple Face detection

cap = cv2.VideoCapture(0) # 0 means for infinite time
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

'''
HOW CASCADE CLASSIFIERS WORKS:
Initially, the algorithm needs a lot of positive images (images of faces) 
and negative images (images without faces) to train the classifier.
Each feature is a single value obtained by subtracting sum of pixels 
under the white rectangle from sum of pixels under the black rectangle.

'''
while True:
    ret, frame = cap.read()
    
    if ret is False:
        continue

    faces = face_cascade.detectMultiScale(frame, scaleFactor = 1.3, minNeighbors=4)
    
    '''
    scaleFactor      Parameter specifying how much the image size is reduced 
                        at each image scale.  

    minNeighbors     Parameter specifying how many neighbors each candidate  
                        rectangle should have to retain it.
    '''

    for face in faces:
        x,y,w,h =  face
        cv2.rectangle(frame, (x,y),(x+w, y+h),(255,255,255),2)


    cv2.imshow('Video Live', frame)
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed is ord('q'):
        break
cap.release()
cv2.destroyAllWindows()