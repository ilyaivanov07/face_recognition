import face_recognition
import cv2

# This is a demo of running face recognition on a video file
#

# Open the input movie file
input_movie = cv2.VideoCapture("ilya.mp4")
if input_movie is None:
    raise Exception("Unable to load movie")

# Load some sample pictures and learn how to recognize them.
ilya1 = face_recognition.load_image_file("../images/ilya.jpg")
ilya1_enc = face_recognition.face_encodings(ilya1)[0]

known_faces = [
    ilya1_enc,
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []

while True:
    # Grab a single frame of video
    ret, frame = input_movie.read()

    # Quit when the input video file ends
    if not ret:
        break

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    # rgb_frame = frame[:, :, ::-1]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        match = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.50)

        name = None
        if match[0]:
            name = "Ilya"

        face_names.append(name)

    # Label the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        if not name:
            continue

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('', frame)

    # quit the program on the press of key 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# All done!
input_movie.release()
cv2.destroyAllWindows()
