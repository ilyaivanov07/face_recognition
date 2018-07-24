import cv2
input_movie = cv2.VideoCapture("hamilton_clip.mp4")

while True:
    # Grab a single frame of video
    ret, frame = input_movie.read()

    # Quit when the input video file ends
    if not ret:
        break

    cv2.imshow('', frame)
    # quit the program on the press of key 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything is done, release the capture
input_movie.release()
cv2.destroyAllWindows()
