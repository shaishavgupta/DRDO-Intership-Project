import face_recognition
from pygame import mixer
import cv2
import numpy as np



video_capture = cv2.VideoCapture(0)

#1
obama_image = face_recognition.load_image_file("obama.jpg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

# 2
archana_image = face_recognition.load_image_file("Archana.jpe")
archana_face_encoding = face_recognition.face_encodings(archana_image)[0]

# 3
shaishav_image = face_recognition.load_image_file("shaishav.jpg")
shaishav_face_encoding = face_recognition.face_encodings(shaishav_image)[0]

            


# arrays
known_face_encodings = [
    obama_face_encoding,
    archana_face_encoding,
    shaishav_face_encoding
]
known_face_names = [
    "Barack Obama",
    "archana",
    "shaishav"
]

while True:
   
    ret, frame = video_capture.read()
    
    rgb_frame = frame[:, :, ::-1]
    
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # match
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        #face_distances = np.linalg.norm(known_face_encodings-face_encoding)

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            # box
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # label
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        else:
            # box
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # label
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            signal(1)
            
            def signal(s):
                mixer.init()
                mixer.music.load("button-6.wav")

                n=5

                while n>s:
                    mixer.music.play()
                    n-=1
        
    
            
        

    # Display
    cv2.imshow('Video', frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video_capture.release()
cv2.destroyAllWindows()

