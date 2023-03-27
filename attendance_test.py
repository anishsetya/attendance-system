
import numpy as np
import cv2 as cv
import pickle
import pandas as pd

df= pd.read_csv('rollnos.csv')

haar_cascade = cv.CascadeClassifier('haar_face.xml')

with open('features.pkl', 'rb') as f:
    features = pickle.load(f)

with open('labels.pkl', 'rb') as f:
    labels = pickle.load(f)

#print(len(features))
#print(labels)
#features=np.array(features,dtype=object)
labels=np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()

face_recognizer.train(features,labels)

def addtofile(x):
    df.loc[x, 'status'] = 'present'




capture=cv.VideoCapture(0)
while True:
    isTrue, frame= capture.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    haar_cascade = cv.CascadeClassifier('haar_face.xml')

    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    for (x,y,w,h) in faces_rect:
        faces_roi = gray[y:y+h,x:x+w]
        label, confidence = face_recognizer.predict(faces_roi)
        #print(f'Label = {people[label]} with a confidence of {confidence}')
        addtofile(label-1)
        cv.putText(frame, str(df.iloc[label-1][2]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
        #print(label)
        cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), thickness=2)
        cv.imshow('Detected Faces', frame)
    if cv.waitKey(20) & 0xFF==ord('d'):
        break

df.to_csv('my_dataframe.csv', index=False)

cv.waitKey(0)   