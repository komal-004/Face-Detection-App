import cv2

# Load the face detection model
face_cascade = cv2.CascadeClassifier(
    r"C:\Users\004ko\Downloads\haarcascade_frontalcatface.xml"
)


# Open webcam (0 means default camera)
camera = cv2.VideoCapture(0)

prev_faces = -1   # add BEFORE while loop

while True:
    ret, frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    # Print only when count changes
    if len(faces) != prev_faces:
        print("Faces detected:", len(faces))
        prev_faces = len(faces)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

    cv2.imshow("Face Detection App", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release camera and close windows
camera.release()
cv2.destroyAllWindows()