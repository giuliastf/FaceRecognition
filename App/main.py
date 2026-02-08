import tkinter as tk
from threading import Thread
from image_upload import upload_and_recognize_image
from image_data_gather import collect_images, get_user_id, global_face_id, upload_training_images
from face_training import train_model
from face_recognizer import recognize_faces
import os

def create_main_ui():
    root = tk.Tk()
    
    root.title("Face Recognition System")
    root.geometry("500x500")
    root.configure(bg='#34e5eb')

    # Use regular tkinter widgets instead of ttk for better color control
    main_frame = tk.Frame(root, bg='#34e5eb', padx=10, pady=10)
    main_frame.pack(fill='both', expand=True)

    # Try to load icon image, but don't crash if it doesn't exist
    icon_image = None
    icon_path = "../Resources/icon.png"
    if os.path.exists(icon_path):
        try:
            icon_image = tk.PhotoImage(file=icon_path)
        except Exception as e:
            print(f"Could not load icon: {e}")
            icon_image = None

    if icon_image:
        title_label = tk.Label(main_frame, text="Face Recognition System", 
                               font=('Helvetica', 18, 'bold'), image=icon_image, 
                               compound='left', bg='#34e5eb', fg='#2c3e50')
        title_label.image = icon_image
    else:
        title_label = tk.Label(main_frame, text="Face Recognition System", 
                               font=('Helvetica', 18, 'bold'), bg='#34e5eb', fg='#2c3e50')
    
    title_label.pack(pady=20)

    button_style = {
        'bg': '#6a1fcc', 
        'fg': '#000000', 
        'font': ('Helvetica', 12, 'bold'), 
        'borderwidth': 0, 
        'relief': 'flat', 
        'activebackground': '#5a0fbc', 
        'activeforeground': '#000000', 
        'cursor': 'hand2',
        'highlightthickness': 0,
        'padx': 20,
        'pady': 10,
        'disabledforeground': '#666666'
    }

    collect_button = tk.Button(main_frame, text="Collect Images (Camera)", command=lambda: (get_user_id(), collect_images()), **button_style)
    collect_button.pack(pady=8)

    upload_button_training = tk.Button(main_frame, text="Upload Training Images", command=upload_training_images, **button_style)
    upload_button_training.pack(pady=8)

    train_button = tk.Button(main_frame, text="Train Model", command=lambda: Thread(target=train_model).start(), **button_style)
    train_button.pack(pady=8)

    recognize_button = tk.Button(main_frame, text="Recognize Faces", command=recognize_faces, **button_style)
    recognize_button.pack(pady=8)

    upload_button = tk.Button(main_frame, text="Upload and Scan Image", command=upload_and_recognize_image, **button_style)
    upload_button.pack(pady=8)

    root.mainloop()

if __name__ == "__main__":
    create_main_ui()
