import face_recognition
import cv2

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
ilya1 = face_recognition.load_image_file("../images/ilya.jpg")
j1 = face_recognition.load_image_file("../images/j1.jpg")
j2 = face_recognition.load_image_file("../images/j2.jpg")
j3 = face_recognition.load_image_file("../images/j3.jpg")
a1 = face_recognition.load_image_file("../images/a1.jpg")
a2 = face_recognition.load_image_file("../images/a2.jpg")
a3 = face_recognition.load_image_file("../images/a3.jpg")
s1 = face_recognition.load_image_file("../images/stephie.jpg")

ilya1_enc = face_recognition.face_encodings(ilya1)[0]
j1_enc = face_recognition.face_encodings(j1)[0]
j2_enc = face_recognition.face_encodings(j2)[0]
j3_enc = face_recognition.face_encodings(j3)[0]
a1_enc = face_recognition.face_encodings(a1)[0]
a2_enc = face_recognition.face_encodings(a2)[0]
a3_enc = face_recognition.face_encodings(a3)[0]
s1_enc = face_recognition.face_encodings(s1)[0]

# Initialize some variables
face_locations = []
face_encodings = []
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)
        face_names = []

        # Loop through each face in this frame
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            match = face_recognition.compare_faces([ilya1_enc, j1_enc, j2_enc, j3_enc, a1_enc, a2_enc, a3_enc, s1_enc], face_encoding)
            if match[0]:
                name = "Ilya"
            elif match[1] or match[2] or match[3]:
                name = "Jackie"
            elif match[4] or match[5] or match[6]:
                name = "Mommy"
            elif match[7]:
                name = "Stephie"
            else:
                name = "Unknown"
            face_names.append(name)

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Draw a box around the face
        cv2.rectangle(small_frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(small_frame, (left, bottom - 15), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(small_frame, name, (left + 3, bottom - 3), font, 0.5, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', small_frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video_capture.release()
cv2.destroyAllWindows()
