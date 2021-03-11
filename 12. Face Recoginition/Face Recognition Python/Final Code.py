
import cv2
import numpy as np 
import os



"""
# Recognize faces using some classifications algorithms - like Logistics, KNN, SVM etc.

1. Load the training data (numpy arrays of all persons)
        # x - values are stored in the numpy arrays
        # y - values we need to assign for each person
2. Read a video stream using openCV
3. Extract faces out of it (testing purposes)
4. Use KNN to find the prediction of face(int)
5. Map the predicted id to name of the user
6. Display the predictions on the screen - bounding box and name

"""

# KNN 

def distance(x1,x2):
    return np.sqrt(sum((x1-x2)**2))  

def knn(train, test, k = 5):
    dist = []
    
    for i in range(train.shape[0]):
        # Get the vector and Labels
        ix = train[i, :-1]
        iy = train[i, -1]
        #Compute the distance from test point
        d = distance(test, ix)
        dist.append([d, iy])
    #Sort based on distance from test point
    dk = sorted(dist, key = lambda x:x[0])[:k]
    #retrieve only the labels
    labels = np.array(dk)[:, -1]
    # Get frequency of each labels
    output  = np.unique(labels, return_counts = True)
    # Find max frequency and corresponding label
    index = np.argmax(output[1])
    return output[0][index]


# Init Camera
cap = cv2.VideoCapture(0)

# Face Detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
skip = 0
dataset_path='./Faces Datasets/'
face_data = list()
labels = list()

class_id = 0
names = {} #mapping btw id and name


# Data Prepration
for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        names[class_id] = fx[:-4]
        print('Loaded: '+fx)
        data_item = np.load(dataset_path+fx, allow_pickle= True)
        face_data.append(data_item)
        
        #create labels for the class
        target = class_id*np.ones((data_item.shape[0],))
        class_id+=1
        labels.append(target)

face_dataset = np.concatenate(face_data, axis = 0)
face_labels = np.concatenate(labels, axis = 0).reshape((-1,1))

print(face_dataset.shape)
print(face_labels.shape)

trainset = np.concatenate((face_dataset, face_labels), axis = 1)
print(trainset.shape)
# Testing Part 

while True:
    ret, frame = cap.read()
    
    if ret == False:
        continue
    # Grayscaled
    faces = face_cascade.detectMultiScale(frame,1.3,5)
    for face in faces:
        x,y,w,h = face
        # Get the face of ROI
        offset = 10
        face_section = frame[y - offset:y+h+offset, x - offset:x+w+offset]
        face_section = cv2.resize(face_section,(100,100))
        # predicted label (out)
        out = knn(trainset,face_section.flatten())
        
        # Display on the screen the name and rectangle around it
        pred_name = names[int(out)]
        cv2.putText(frame, pred_name,(x,y-10), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),2)
    cv2.imshow('faces', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()