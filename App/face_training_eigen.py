"""
Eigenfaces (PCA) Face Recognition Training Module

This module implements PCA-based face recognition using Eigenfaces approach.
Each person gets their own PCA subspace computed from their training images.
Recognition is based on reconstruction error in each person's subspace.
"""

import os
import cv2
import numpy as np
import pickle
from PIL import Image
from sklearn.decomposition import PCA
from tkinter import messagebox


class EigenFaceModel:
    """Eigenfaces-based face recognition model using PCA"""
    
    def __init__(self, variance_threshold=0.95, image_size=(100, 100)):
        """
        Initialize EigenFace model
        
        Args:
            variance_threshold: Percentage of variance to retain (default 0.95 = 95%)
            image_size: Size to resize all faces to (width, height)
        """
        self.variance_threshold = variance_threshold
        self.image_size = image_size
        self.person_models = {}  # Dictionary to store PCA models for each person
        self.person_means = {}   # Mean faces for each person
        self.labels_dict = {}    # person_name -> person_id mapping
        
    def train_person_model(self, person_id, person_name, face_vectors):
        """
        Train PCA model for a specific person
        
        Args:
            person_id: Numeric ID for the person
            person_name: Name of the person
            face_vectors: Array of flattened face images (n_samples, n_features)
        """
        if len(face_vectors) < 2:
            print(f"[WARNING] Person {person_name} has only {len(face_vectors)} images. Need at least 2 for PCA.")
            return False
            
        print(f"[INFO] Training Eigenfaces for {person_name} (ID={person_id})...")
        
        X = np.array(face_vectors)
        
        # Compute mean face for this person
        mean_face = np.mean(X, axis=0)
        self.person_means[person_id] = mean_face
        
        # Center the data (subtract mean)
        X_centered = X - mean_face
        
        # Apply PCA
        pca = PCA()
        pca.fit(X_centered)
        
        # Determine number of components based on variance threshold
        cumsum_var = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.argmax(cumsum_var >= self.variance_threshold) + 1
        n_components = min(n_components, len(face_vectors) - 1)  # Can't exceed samples - 1
        
        # Keep only the required number of components
        pca_final = PCA(n_components=n_components)
        pca_final.fit(X_centered)
        
        # Store the model
        self.person_models[person_id] = {
            'pca': pca_final,
            'mean_face': mean_face,
            'n_components': n_components,
            'explained_variance': pca_final.explained_variance_ratio_.sum(),
            'person_name': person_name
        }
        
        print(f"    ✓ Components: {n_components} (variance: {pca_final.explained_variance_ratio_.sum():.3f})")
        return True
    
    def predict(self, test_vector):
        """
        Predict the person ID for a test face
        
        Args:
            test_vector: Flattened face image vector
            
        Returns:
            (person_id, confidence_score, reconstruction_errors)
        """
        if not self.person_models:
            return -1, 100.0, {}
            
        reconstruction_errors = {}
        
        for person_id, model in self.person_models.items():
            pca = model['pca']
            mean_face = model['mean_face']
            
            # Center the test vector
            test_centered = test_vector - mean_face
            
            # Project onto the person's eigenface subspace
            projection = pca.transform(test_centered.reshape(1, -1))
            
            # Reconstruct the image
            reconstruction = pca.inverse_transform(projection)
            reconstruction = reconstruction.flatten() + mean_face
            
            # Calculate reconstruction error (Euclidean distance)
            error = np.linalg.norm(test_vector - reconstruction)
            reconstruction_errors[person_id] = error
        
        # Return the person with minimum reconstruction error
        predicted_person = min(reconstruction_errors, key=reconstruction_errors.get)
        min_error = reconstruction_errors[predicted_person]
        
        # Convert error to confidence (0-100, lower error = higher confidence)
        # Normalize error to confidence scale
        confidence = min(min_error, 100.0)
        
        return predicted_person, confidence, reconstruction_errors
    
    def get_eigenfaces(self, person_id, max_components=5):
        """
        Get eigenfaces (principal components) for visualization
        
        Args:
            person_id: ID of the person
            max_components: Maximum number of eigenfaces to return
            
        Returns:
            (mean_face, eigenfaces_list) as 2D images
        """
        if person_id not in self.person_models:
            return None, []
            
        model = self.person_models[person_id]
        pca = model['pca']
        mean_face = model['mean_face']
        
        # Reshape mean face to 2D
        mean_face_img = mean_face.reshape(self.image_size)
        
        # Get principal components (eigenfaces)
        n_comp = min(max_components, model['n_components'])
        eigenfaces = []
        
        for i in range(n_comp):
            eigenface = pca.components_[i]
            # Normalize for visualization
            eigenface_norm = (eigenface - eigenface.min()) / (eigenface.max() - eigenface.min())
            eigenface_img = eigenface_norm.reshape(self.image_size)
            eigenfaces.append(eigenface_img)
        
        return mean_face_img, eigenfaces


def train_eigenfaces_model(dataset_path='dataset', output_path='trainer'):
    """
    Train Eigenfaces model on all people in the dataset
    
    Args:
        dataset_path: Path to dataset folder with person subfolders
        output_path: Path to save trained model
        
    Returns:
        True if successful, False otherwise
    """
    print("\n" + "="*60)
    print("TRAINING EIGENFACES (PCA) MODEL")
    print("="*60)
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        messagebox.showerror("Error", "No dataset folder found. Please collect images first.")
        return False
    
    # Get list of people
    people = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    if not people:
        messagebox.showerror("Error", "No people found in dataset. Please collect images first.")
        return False
    
    print(f"[INFO] Found {len(people)} people: {people}")
    
    # Initialize model
    model = EigenFaceModel(variance_threshold=0.95, image_size=(100, 100))
    
    # Load face detector
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    # Build labels dictionary
    labels_dict = {name: idx for idx, name in enumerate(sorted(people))}
    model.labels_dict = labels_dict
    
    print(f"[INFO] Label mapping: {labels_dict}")
    
    # Process each person
    for person_name in sorted(people):
        person_id = labels_dict[person_name]
        person_path = os.path.join(dataset_path, person_name)
        
        # Load all images for this person
        face_vectors = []
        image_files = [f for f in os.listdir(person_path) if f.endswith('.jpg')]
        
        print(f"\n[INFO] Processing {person_name}: {len(image_files)} images")
        
        for img_file in image_files:
            img_path = os.path.join(person_path, img_file)
            
            # Load image
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            
            if len(faces) == 0:
                # If no face detected, use entire image
                face_region = gray
            else:
                # Use first detected face
                x, y, w, h = faces[0]
                face_region = gray[y:y+h, x:x+w]
            
            # Resize to standard size
            face_resized = cv2.resize(face_region, model.image_size)
            
            # Normalize to [0, 1]
            face_normalized = face_resized.astype(np.float64) / 255.0
            
            # Flatten to vector
            face_vector = face_normalized.flatten()
            face_vectors.append(face_vector)
        
        # Train model for this person
        if len(face_vectors) >= 2:
            model.train_person_model(person_id, person_name, face_vectors)
        else:
            print(f"[WARNING] Skipping {person_name} - insufficient images")
    
    # Save model
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    model_path = os.path.join(output_path, 'trainer_eigen.pkl')
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"\n[SUCCESS] Eigenfaces model saved to {model_path}")
    print(f"[INFO] Trained for {len(model.person_models)} people")
    print("="*60 + "\n")
    
    messagebox.showinfo("Training Complete", 
                       f"Eigenfaces model trained successfully!\n\n"
                       f"People: {len(model.person_models)}\n"
                       f"Model saved to: {model_path}")
    
    return True


def load_eigenfaces_model(model_path='trainer/trainer_eigen.pkl'):
    """
    Load trained Eigenfaces model
    
    Args:
        model_path: Path to saved model file
        
    Returns:
        EigenFaceModel instance or None if loading fails
    """
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found: {model_path}")
        return None
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"[INFO] Eigenfaces model loaded from {model_path}")
        return model
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return None


if __name__ == "__main__":
    # Train model when run directly
    train_eigenfaces_model()
