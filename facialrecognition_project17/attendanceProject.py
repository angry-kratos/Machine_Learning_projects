import cv2
import numpy as np
import face_recognition
import os


path = 'trainimages'
images=[]
classNames = []
myList = os.listdir(path)

print(myList)


attendance = []


# firstly, we will find all the faces in the given attendace pic
attendancepic = face_recognition.load_image_file('trainimages\\avengers.jpg')
image_attendance  = cv2.cvtColor(attendancepic, cv2.COLOR_BGR2RGB)

face_locations = face_recognition.face_locations(image_attendance)
face_encoding_attendance = face_recognition.face_encodings(image_attendance)

print(face_locations)

for i in range(len(face_locations)):
    image_attendance = cv2.rectangle(image_attendance, (face_locations[i][3], face_locations[i][0]), (face_locations[i][1], face_locations[i][2]), (255, 0 , 255), 2)
    img_copy = image_attendance
    face  = img_copy[face_locations[i][0]:face_locations[i][2],face_locations[i][3]: face_locations[i][1]]
    img_copy = image_attendance
    cv2.imwrite(f'face_{i}.jpg', face)




cv2.imshow('all faces', image_attendance)

cv2.waitKey(0)



images=[]
classNames = []

for cls in myList:
    curImg = cv2.imread(f'{path}\\{cls}')
    images.append(curImg)
    classNames.append(os.path.splitext(cls)[0])

print(classNames)

def findencodings(images):
    encodeList = []
    for img in images:
        img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)
        encodeList.append(encode)
    return encodeList

encodeListKnown = findencodings(images)

#print(face_recognition.compare_faces(encodeListKnown[5], face_encoding_attendance[10]))
print(len(encodeListKnown))

for attender in face_encoding_attendance:
    for known_faces_no in range(len(encodeListKnown)):
        results = face_recognition.compare_faces(encodeListKnown[known_faces_no], attender)
        if results == [True]:
            attendance.append(classNames[known_faces_no])

print(attendance)
cv2.waitKey(0)

