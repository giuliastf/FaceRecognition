import cv2

# Load the classifier
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Prepering the camera
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # set Width
cam.set(4, 480)  # set Height

# For each person, enter one numeric face id (labeling data)
face_id = input("\n enter user id end press <return> ==>  ")
print("\n [INFO] Initializing face capture. Look the camera and wait ...")

count = 0
while True:
    ret, img = cam.read()

    # Grayscaling the image for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Calling the classifier function
    faces = face_cascade.detectMultiScale(
        gray,  # Gray sclae the imag
        scaleFactor=1.2,  # How much the image size is reduced
        minNeighbors=5,  # Number of neighbors each candidate rectangle should have
        minSize=(20, 20),  # The minimum rectangle size to be considered a face
    )

    for x, y, w, h in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        count += 1

        # Save the captured image into the datasets folder
        cv2.imwrite(
            "dataset/User." + str(face_id) + "." + str(count) + ".jpg",
            gray[y : y + h, x : x + w],
        )
        cv2.imshow("image", img)

        # roi_gray = gray[y : y + h, x : x + w]
        # roi_color = img[y : y + h, x : x + w]

    # Exit functionality
    k = cv2.waitKey(30) & 0xFF
    if k == 27:  # press 'ESC' to quit
        break
    elif count >= 150:  # Take 30 face sample and then stop
        break

# Cleanup
cam.release()
cv2.destroyAllWindows()
