import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

video_capture = cv2.VideoCapture(0)

utkarsh = face_recognition.load_image_file("utkarsh.jpg")
utkarsh_encoding = face_recognition.face_encodings(utkarsh)[0]

dino = face_recognition.load_image_file("dino.jpg")
dino_encoding = face_recognition.face_encodings(dino)[0]

dhruv = face_recognition.load_image_file("dhruv.jpg")
dhruv_encoding = face_recognition.face_encodings(dhruv)[0]

priyansh = face_recognition.load_image_file("priyansh.jpg")
priyansh_encoding = face_recognition.face_encodings(priyansh)[0]

uriel = face_recognition.load_image_file("uriel.jpg")
uriel_encoding = face_recognition.face_encodings(uriel)[0]

javedsir = face_recognition.load_image_file("javedsir.jpg")
javedsir_encoding = face_recognition.face_encodings(javedsir)[0]



known_faces_names = ["utkarsh","dino","dhruv","priyansh","uriel","javedsir"]
known_faces_encoding = [utkarsh_encoding,dino_encoding,dhruv_encoding,priyansh_encoding,uriel_encoding,javedsir_encoding]

students = ["utkarsh","dino","dhruv","priyansh","uriel","javedsir"]

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
current_date = datetime.now().strftime("%Y-%m-%d")
f = open(current_date+'.csv','w+',newline = '')
lnwriter = csv.writer(f)

while True:
    ret, frame = video_capture.read()
    small_frame = frame
    rgb_small_frame = small_frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    face_names = []

    print("Number of faces detected:", len(face_locations))

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_faces_encoding, face_encoding)
        name = ""
        face_distance = face_recognition.face_distance(known_faces_encoding, face_encoding)
        best_match_index = np.argmin(face_distance)

        if matches[best_match_index]:
            name = known_faces_names[best_match_index]

        if name:
            if name in students:
                students.remove(name)
                print("Student removed:", name)
                current_time = now.strftime("%H:%M:%S")
                lnwriter.writerow([name, str(current_time)])
                print(str(current_time))
            #else:
                #print("Unknown face detected:", name)

        face_names.append(name)

    print("Detected face names:", face_names)
    print("Remaining students:", students)

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1)

    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
f.close()
video_capture.release()

cv2.destroyAllWindows()
