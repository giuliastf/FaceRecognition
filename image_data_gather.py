import cv2
import os
import tkinter as tk
from tkinter import simpledialog, messagebox

global_face_id = None

def get_user_id():
    global global_face_id
    global_face_id = simpledialog.askstring("Input", "Who is this?")
    return global_face_id

def collect_images():
    global global_face_id
    face_id = global_face_id
    if not face_id:
        print("No user id entered. Exiting...")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture")
        return

    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if face_detector.empty():
        print("Error: Could not load Haar cascade")
        return

    dataset_path = os.path.join(r"FaceRecognition", 'dataset', face_id)
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
        print(f"[INFO] Created directory: {dataset_path}")
    else:
        print(f"[INFO] Directory already exists: {dataset_path}")

    count = 0
    while True:
        ret, img = cap.read()
        if not ret:
            print("Error: Could not read frame from video capture")
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            count += 1
            image_path = os.path.join(dataset_path, f"{str(count)}.jpg")
            cv2.imwrite(image_path, gray[y:y + h, x:x + w])
            print(f"[INFO] Saved image: {image_path}")
            cv2.imshow('image', img)

        k = cv2.waitKey(100) & 0xff  # Press 'ESC' for exiting video
        if k == 27 or count >= 30:  # Take 30 face samples and stop video
            break

    cap.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("Info", f"Collected {count} images for user {face_id}")
