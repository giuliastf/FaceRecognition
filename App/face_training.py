import os
import cv2
from PIL import Image
import numpy as np
import pickle
from tkinter import messagebox
from face_training_eigen import train_eigenfaces_model

def train_model():
    """Train both LBPH and Eigenfaces models"""
    print("\n" + "="*60)
    print("MULTI-ALGORITHM TRAINING: LBPH + EIGENFACES")
    print("="*60)
    
    # Train LBPH model
    success_lbph = train_lbph_model()
    
    # Train Eigenfaces model
    success_eigen = train_eigenfaces_model()
    
    # Show combined result
    if success_lbph and success_eigen:
        messagebox.showinfo("Training Complete", 
                           "Both models trained successfully!\n\n"
                           "✓ LBPH (Local Binary Patterns)\n"
                           "✓ Eigenfaces (PCA)\n\n"
                           "You can now compare both algorithms.")
    elif success_lbph:
        messagebox.showwarning("Partial Success", 
                              "LBPH model trained successfully.\n"
                              "Eigenfaces training failed.")
    elif success_eigen:
        messagebox.showwarning("Partial Success", 
                              "Eigenfaces model trained successfully.\n"
                              "LBPH training failed.")
    else:
        messagebox.showerror("Training Failed", 
                            "Both models failed to train.")

def train_lbph_model():
    """Train LBPH (Local Binary Patterns Histogram) model"""
    print("\n[LBPH] Starting LBPH training...")
    dataset_path = os.path.join('dataset')
    trainer_path = os.path.join('trainer', 'trainer_lbph.yml')
    labels_path = os.path.join('trainer', 'labels.pickle')

    # Check if dataset folder exists and has content
    if not os.path.exists(dataset_path):
        messagebox.showerror("Error", "No dataset folder found. Please collect images first.")
        return False
    
    # List available people in dataset
    people = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    if not people:
        messagebox.showerror("Error", "No people found in dataset. Please collect images first.")
        return False
    
    print(f"[LBPH] Training with people: {people}")

    # Create LBPH recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    if os.path.exists(trainer_path) and os.path.exists(labels_path):
        recognizer.read(trainer_path)
        with open(labels_path, 'rb') as f:
            labels_dict = pickle.load(f)
        current_id = max(labels_dict.values()) + 1
    else:
        labels_dict = {}
        current_id = 0

    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def getImagesAndLabels(path, current_id, labels_dict):
        faceSamples = []
        ids = []
        
        # Create CLAHE object for preprocessing  
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        print("[LBPH] Using CLAHE preprocessing for better lighting normalization")

        for root, dirs, files in os.walk(path):
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                if dir_name not in labels_dict:
                    labels_dict[dir_name] = current_id
                    current_id += 1
                id_ = labels_dict[dir_name]
                for file in os.listdir(dir_path):
                    if file.endswith(".jpg"):
                        imagePath = os.path.join(dir_path, file)
                        PIL_img = Image.open(imagePath).convert("L")  # grayscale
                        img_numpy = np.array(PIL_img, "uint8")
                        faces = detector.detectMultiScale(img_numpy)

                        for (x, y, w, h) in faces:
                            face_crop = img_numpy[y:y + h, x:x + w]
                            
                            # Apply CLAHE preprocessing
                            face_enhanced = clahe.apply(face_crop)
                            
                            # Resize to consistent dimensions  
                            face_resized = cv2.resize(face_enhanced, (150, 150))
                            
                            faceSamples.append(face_resized)
                            ids.append(id_)

        return faceSamples, ids

    faces, ids = getImagesAndLabels(dataset_path, current_id, labels_dict)
    recognizer.train(faces, np.array(ids))
    if not os.path.exists(os.path.dirname(trainer_path)):
        os.makedirs(os.path.dirname(trainer_path))
    recognizer.write(trainer_path)

    with open(labels_path, 'wb') as f:
        pickle.dump(labels_dict, f)

    print(f"[LBPH] Model saved to {trainer_path}")
    print(f"[LBPH] Labels saved to {labels_path}")
    print(f"[LBPH] Training complete for {len(people)} people")
    
    return True
