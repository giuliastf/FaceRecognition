import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import os

def upload_and_recognize_image():
    print("[DEBUG] upload_and_recognize_image called")
    # Open file dialog to select image
    file_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
    )
    
    print(f"[DEBUG] Selected file: {file_path}")
    
    if not file_path:
        print("[DEBUG] No file selected")
        return
    
    try:
        print("[DEBUG] Entering try block")
        # Load the trained model
        trainer_path = os.path.join('trainer', 'trainer.yml')
        labels_path = os.path.join('trainer', 'labels.pickle')
        print(f"[DEBUG] Trainer path: {trainer_path}, exists: {os.path.exists(trainer_path)}")
        print(f"[DEBUG] Labels path: {labels_path}, exists: {os.path.exists(labels_path)}")
        if not os.path.exists(trainer_path):
            messagebox.showerror("Error", "No trained model found. Please train the model first.")
            return
            
        print("[DEBUG] Creating recognizer")
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        print("[DEBUG] Recognizer created, now reading model file")
        recognizer.read(trainer_path)
        print("[DEBUG] Model loaded successfully")
        
        # Load the name labels
        labels_dict = {}
        if os.path.exists(labels_path):
            import pickle
            with open(labels_path, 'rb') as f:
                labels_dict = pickle.load(f)
                labels_dict = {v: k for k, v in labels_dict.items()}  # Invert: id -> name
        
        # Load face cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Create CLAHE object for preprocessing (same as training)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        
        # Load and process the image
        img = cv2.imread(file_path)
        if img is None:
            messagebox.showerror("Error", "Could not load image file.")
            return
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces with improved parameters for better detection
        # scaleFactor=1.1: smaller steps = more thorough detection
        # minNeighbors=3: less strict = detect more faces
        # minSize=(30,30): ignore very small detections (noise)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
        
        if len(faces) == 0:
            messagebox.showinfo("Result", "No faces detected in the image.")
            return
        
        # Recognize faces
        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = gray[y:y+h, x:x+w]
            
            # Apply same preprocessing as training
            face_enhanced = clahe.apply(face_roi)
            face_resized = cv2.resize(face_enhanced, (150, 150))
            
            id_, confidence = recognizer.predict(face_resized)
            
            print(f"[DEBUG] Recognition result: ID={id_}, Confidence={confidence}")
            print(f"[DEBUG] Available labels: {labels_dict}")
            
            # Draw rectangle and label
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Handle ID=-1 (no match found)
            if id_ == -1:
                label = "Unknown Person"
                color = (0, 0, 255)  # Red
                print(f"[DEBUG] Face not in training data (ID=-1)")
            elif confidence < 80:  # High confidence - good match
                name = labels_dict.get(id_, f"Unknown_ID_{id_}")
                label = f"{name} ({confidence:.1f})"
                color = (0, 255, 0)  # Green for confident recognition
                print(f"[DEBUG] Recognized as: {name} (HIGH CONFIDENCE)")
            elif confidence < 100:  # Medium confidence - likely correct but uncertain
                name = labels_dict.get(id_, f"Unknown_ID_{id_}")
                label = f"{name}? ({confidence:.1f})"
                color = (0, 165, 255)  # Orange for uncertain
                print(f"[DEBUG] Recognized as: {name} (UNCERTAIN)")
            else:  # Low confidence - probably wrong
                label = f"Unknown (conf: {confidence:.1f})"
                color = (0, 0, 255)  # Red for unknown
                print(f"[DEBUG] Not recognized - confidence too low: {confidence}")
                
            cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Display result
        cv2.imshow('Face Recognition Result', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        messagebox.showinfo("Success", f"Found {len(faces)} face(s) in the image.")
        
    except Exception as e:
        print(f"[ERROR] Exception occurred: {e}")
        import traceback
        traceback.print_exc()
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

def get_user_id():
    user_id = simpledialog.askstring("Input", "Who is this?")
    return user_id

def save_detected_face(image, face_location):
    top, right, bottom, left = face_location
    face_image = image[top:bottom, left:right]

    # Convert the face image to a PIL Image
    gray_face_image = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
    pil_image = Image.fromarray(gray_face_image)
    
    # Prompt for user ID
    user_id = get_user_id()
    if not user_id:
        messagebox.showinfo("Result", "No User ID provided. Skipping this face.")
        return
    else:
        image_window = tk.Toplevel()
        image_window.title("Detected Face")
        image_window.geometry("300x300")

        # Convert the PIL Image to a Tkinter-compatible image
        tk_image = ImageTk.PhotoImage(pil_image)

        panel = tk.Label(image_window, image=tk_image)
        panel.image = tk_image
        panel.pack()

        # Close the window after displaying
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
