import cv2 as cv
import numpy as np
import pickle

capture=cv.VideoCapture(0)
haar_cascade = cv.CascadeClassifier('haar_face.xml')
def face_extractor(img):
    # Function detects faces and returns the cropped face
    # If no face detected, it returns the input image
    
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7)
    
    if faces_rect is ():
        return None
    
    # Crop all faces found
    for (x,y,w,h) in faces_rect:
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face



with open('features.pkl', 'rb') as f:
    features = pickle.load(f)

with open('labels.pkl', 'rb') as f:
    labels = pickle.load(f)


print(len(labels))
print(len(features))

def create_train(i):
    count = 0
    while True:
        ret, frame = capture.read()
        if face_extractor(frame) is not None:
            count += 1
            face = cv.resize(face_extractor(frame), (200, 200))
            face = cv.cvtColor(face, cv.COLOR_BGR2GRAY)

            features.append(face)
            labels.append(i)

            # Put count on images and display live count
            cv.putText(face, str(count), (50, 50), cv.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 4)
            cv.imshow('Face Cropper', face)
            
        else:
            print("Face not found")
            pass

        if ( cv.waitKey(20) & 0xFF==ord('d'))  or count == 100: #d is the Enter Key
            break

i=int(input("Enter roll number"))
create_train(i)
# features = np.array(features, dtype='object')

with open('features.pkl', 'wb') as f:
    pickle.dump(features, f)

with open('labels.pkl', 'wb') as f:
    pickle.dump(labels, f)

