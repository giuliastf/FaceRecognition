import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
import face_recognition
from PIL import Image, ImageTk
import os
import cv2

def get_user_id():
    user_id = simpledialog.askstring("Input", "Who is this?")
    return user_id

def upload_and_recognize_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if not file_path:
        return

    try:
        image = face_recognition.load_image_file(file_path)
        face_locations = face_recognition.face_locations(image)

        if not face_locations:
            messagebox.showinfo("Result", "No faces found in the uploaded image.")
        else:
            messagebox.showinfo("Result", f"Found {len(face_locations)} face(s) in the uploaded image.")
            for face_location in face_locations:
                save_detected_face(image, face_location)

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

def save_detected_face(image, face_location):
    top, right, bottom, left = face_location
    face_image = image[top:bottom, left:right]

    # Convert the face image to a PIL Image
    gray_face_image = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
    pil_image = Image.fromarray(gray_face_image)
    
    # Prompt for user ID
    

    

    # Create a new window for each face
    image_window = tk.Toplevel()
    image_window.title("Detected Face")
    image_window.geometry("300x300")

    # Convert the PIL Image to a Tkinter-compatible image
    tk_image = ImageTk.PhotoImage(pil_image)

    panel = tk.Label(image_window, image=tk_image)
    panel.image = tk_image
    panel.pack()

    # Close the window after displaying
    # image_window.after(5000, image_window.destroy)
    user_id = get_user_id()
    if not user_id:
        messagebox.showinfo("Result", "No User ID provided. Skipping this face.")
        return
    else:
        image_window.after(5000, image_window.destroy)
    dataset_path = os.path.join("FaceRecognition", "dataset", user_id)
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    # Save the face image
    face_filename = os.path.join(dataset_path, f"{len(os.listdir(dataset_path)) + 1}.jpg")
    pil_image.save(face_filename)
    print(f"[INFO] Saved image: {face_filename}")

# def display_image(image_path):
#     image_window = tk.Toplevel()
#     image_window.title("Uploaded Image")
#     image_window.geometry("600x600")

#     img = Image.open(image_path)
#     img = img.resize((500, 500), Image.ANTIALIAS)
#     img = ImageTk.PhotoImage(img)

#     panel = tk.Label(image_window, image=img)
#     panel.image = img
#     panel.pack()
