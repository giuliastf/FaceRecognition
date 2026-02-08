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
        
        # Calculate adaptive text scale based on image dimensions
        img_height, img_width = img.shape[:2]
        base_scale = min(img_width, img_height) / 1000.0  # Scale factor based on image size
        font_scale = max(0.4, min(1.2, base_scale))  # Keep between 0.4 and 1.2
        font_thickness = max(1, int(font_scale * 2))
        
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
            
            # Calculate adaptive rectangle thickness based on face size
            rect_thickness = max(1, int(w / 100))
            
            # Draw rectangle and label
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), rect_thickness)
            
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
            
            # Get text size for positioning and background
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 
                                                                   font_scale, font_thickness)
            
            # Position text above face, with background for better readability
            text_x = x
            text_y = y - 10
            
            # If text would go above image, put it below the top of rectangle
            if text_y - text_height < 0:
                text_y = y + text_height + 10
            
            # Draw background rectangle for text
            cv2.rectangle(img, (text_x, text_y - text_height - 5), 
                         (text_x + text_width + 5, text_y + 5), (0, 0, 0), -1)
            
            # Draw text
            cv2.putText(img, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                       font_scale, color, font_thickness)
        
        # Add instruction text at the bottom of the image
        h_img, w_img = img.shape[:2]
        instruction_text = "Press any key or click 'X' to close"
        text_size = cv2.getTextSize(instruction_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = (w_img - text_size[0]) // 2
        text_y = h_img - 20
        
        # Add a background rectangle for better visibility
        cv2.rectangle(img, (text_x - 10, text_y - text_size[1] - 10), 
                     (text_x + text_size[0] + 10, text_y + 10), (0, 0, 0), -1)
        cv2.putText(img, instruction_text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display result
        window_name = 'Face Recognition Result'
        cv2.imshow(window_name, img)
        
        # Wait for key press or window close
        while True:
            key = cv2.waitKey(100)  # Check every 100ms
            if key != -1:  # Any key pressed
                break
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:  # Window closed
                break
        
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
