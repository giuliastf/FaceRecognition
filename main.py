import tkinter as tk
from tkinter import ttk
from threading import Thread
from image_upload import upload_and_recognize_image
from style import apply_styles 
from image_data_gather import collect_images, get_user_id
from face_training import train_model
from face_recognizer import recognize_faces

def start_recognition_thread():
    Thread(target=recognize_faces).start()

def create_main_ui():
    root = tk.Tk()
    
    root.title("Face Recognition System")
    root.geometry("500x500")  # Increase the height to 500
    root.configure(bg='#34e5eb')  # Change background color to a slightly darker shade of blue

    apply_styles(root)

    main_frame = ttk.Frame(root, padding="10", style="TFrame")
    main_frame.pack(fill='both', expand=True)

    # Load an icon image
    icon_image = tk.PhotoImage(file=r"FaceRecognition/icon.png")   # Update the path to your icon image

    title_label = ttk.Label(main_frame, text="Face Recognition System", font=('Helvetica', 18, 'bold'), image=icon_image, compound='left', style="TLabel")
    title_label.image = icon_image  # Keep a reference to the image to avoid garbage collection
    title_label.pack(pady=20)

    button_style = {'bg': '#6a1fcc', 'fg': 'white', 'font': ('Helvetica', 12), 'borderwidth': 1, 'relief': 'solid'}

    collect_button = tk.Button(main_frame, text="Collect Images", command=lambda: [get_user_id(), Thread(target=collect_images).start()], **button_style)
    collect_button.pack(pady=10, ipadx=10, ipady=5)

    train_button = tk.Button(main_frame, text="Train Model", command=lambda: Thread(target=train_model).start(), **button_style)
    train_button.pack(pady=10, ipadx=10, ipady=5)

    recognize_button = tk.Button(main_frame, text="Recognize Faces", command=start_recognition_thread, **button_style)
    recognize_button.pack(pady=10, ipadx=10, ipady=5)

    upload_button = tk.Button(main_frame, text="Upload and Scan Image", command=upload_and_recognize_image, **button_style)
    upload_button.pack(pady=10, ipadx=10, ipady=5)

    root.mainloop()

if __name__ == "__main__":
    create_main_ui()
