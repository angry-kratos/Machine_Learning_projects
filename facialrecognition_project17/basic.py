import cv2
import numpy as np
import face_recognition

imgRoberttrain = face_recognition.load_image_file('facialrecognition\\trainimages\\htest1.jpeg')
imgRoberttest = face_recognition.load_image_file('facialrecognition\\trainimages\\htrain1.jpeg')
imgRoberttrain = cv2.cvtColor(imgRoberttrain, cv2.COLOR_BGR2RGB)
imgRoberttest = cv2.cvtColor(imgRoberttest, cv2.COLOR_BGR2RGB)

faceLoc  = face_recognition.face_locations(imgRoberttrain)[0]
encodeRobert = face_recognition.face_encodings(imgRoberttrain)[0]
cv2.rectangle(imgRoberttrain, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

faceLocTest  = face_recognition.face_locations(imgRoberttest)[0]
encodeRoberTest = face_recognition.face_encodings(imgRoberttest)[0]
cv2.rectangle(imgRoberttest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)

results = face_recognition.compare_faces([encodeRobert], encodeRoberTest)
distance = face_recognition.face_distance([encodeRobert], encodeRoberTest)
print(results, distance)

cv2.putText(imgRoberttest, f'{results} {np.round(distance,2)}', (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255),2)

cv2.imshow('robert_train_1', imgRoberttrain)
cv2.imshow('robert_test_1', imgRoberttest)
cv2.waitKey(0)
