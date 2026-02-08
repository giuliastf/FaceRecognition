"""
Unified Recognition Module - Supports Both LBPH and Eigenfaces

This module provides a unified interface for face recognition using either:
1. LBPH (Local Binary Patterns Histogram)
2. Eigenfaces (PCA-based)
3. Both algorithms for comparison
"""

import cv2
import numpy as np
import pickle
import os
from face_training_eigen import load_eigenfaces_model


# Global variable to store selected algorithm
selected_algorithm = "LBPH"  # Default: "LBPH", "Eigenfaces", or "Both"


def set_algorithm(algo):
    """Set the active recognition algorithm"""
    global selected_algorithm
    selected_algorithm = algo
    print(f"[INFO] Algorithm changed to: {selected_algorithm}")


def get_algorithm():
    """Get the current recognition algorithm"""
    return selected_algorithm


def recognize_with_lbph(face_roi, clahe):
    """
    Recognize face using LBPH algorithm
    
    Returns: (id, confidence, name, label_text, color)
    """
    # Load LBPH model
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    trainer_path = 'trainer/trainer_lbph.yml'
    
    if not os.path.exists(trainer_path):
        return -1, 100, "No Model", "No LBPH Model", (0, 0, 255)
    
    recognizer.read(trainer_path)
    
    # Load labels
    labels_dict = {}
    labels_path = 'trainer/labels.pickle'
    if os.path.exists(labels_path):
        with open(labels_path, 'rb') as f:
            labels_dict = pickle.load(f)
            labels_dict = {v: k for k, v in labels_dict.items()}
    
    # Preprocess face
    face_enhanced = clahe.apply(face_roi)
    face_resized = cv2.resize(face_enhanced, (150, 150))
    
    # Predict
    id_, confidence = recognizer.predict(face_resized)
    
    # Determine label and color based on confidence
    if id_ == -1:
        name = "Unknown"
        label = "Unknown (LBPH)"
        color = (0, 0, 255)  # Red
    elif confidence < 80:
        name = labels_dict.get(id_, f"Unknown_ID_{id_}")
        label = f"{name} ({confidence:.0f})"
        color = (0, 255, 0)  # Green
    elif confidence < 100:
        name = labels_dict.get(id_, f"Unknown_ID_{id_}")
        label = f"{name}? ({confidence:.0f})"
        color = (0, 165, 255)  # Orange
    else:
        name = "Unknown"
        label = f"Unknown ({confidence:.0f})"
        color = (0, 0, 255)  # Red
    
    return id_, confidence, name, label, color


def recognize_with_eigenfaces(face_roi):
    """
    Recognize face using Eigenfaces (PCA) algorithm
    
    Returns: (id, confidence, name, label_text, color)
    """
    # Load Eigenfaces model
    model_path = 'trainer/trainer_eigen.pkl'
    
    if not os.path.exists(model_path):
        return -1, 100, "No Model", "No Eigen Model", (0, 0, 255)
    
    eigen_model = load_eigenfaces_model(model_path)
    if eigen_model is None:
        return -1, 100, "Error", "Eigen Load Error", (0, 0, 255)
    
    # Preprocess face
    face_resized = cv2.resize(face_roi, eigen_model.image_size)
    face_normalized = face_resized.astype(np.float64) / 255.0
    face_vector = face_normalized.flatten()
    
    # Predict
    id_, confidence, errors = eigen_model.predict(face_vector)
    
    # Get name from model
    if id_ in eigen_model.person_models:
        name = eigen_model.person_models[id_]['person_name']
    else:
        name = "Unknown"
    
    # Determine label and color (Eigenfaces uses reconstruction error)
    # Lower error = better match
    if confidence < 30:
        label = f"{name} ({confidence:.0f})"
        color = (0, 255, 0)  # Green - high confidence
    elif confidence < 50:
        label = f"{name}? ({confidence:.0f})"
        color = (0, 165, 255)  # Orange - medium confidence
    else:
        label = f"Unknown ({confidence:.0f})"
        color = (0, 0, 255)  # Red - low confidence
        name = "Unknown"
    
    return id_, confidence, name, label, color


def recognize_face_unified(face_roi, clahe, algorithm=None):
    """
    Unified face recognition function that supports multiple algorithms
    
    Args:
        face_roi: Face region (grayscale)
        clahe: CLAHE object for preprocessing
        algorithm: "LBPH", "Eigenfaces", or "Both" (uses global if None)
        
    Returns:
        For single algorithm: (id, confidence, name, label, color)
        For "Both": (results_dict, combined_label, combined_color)
    """
    if algorithm is None:
        algorithm = get_algorithm()
    
    if algorithm == "LBPH":
        return recognize_with_lbph(face_roi, clahe)
    
    elif algorithm == "Eigenfaces":
        return recognize_with_eigenfaces(face_roi)
    
    elif algorithm == "Both":
        # Run both algorithms and return comparison
        lbph_results = recognize_with_lbph(face_roi, clahe)
        eigen_results = recognize_with_eigenfaces(face_roi)
        
        results = {
            'LBPH': {
                'id': lbph_results[0],
                'confidence': lbph_results[1],
                'name': lbph_results[2],
                'label': lbph_results[3],
                'color': lbph_results[4]
            },
            'Eigenfaces': {
                'id': eigen_results[0],
                'confidence': eigen_results[1],
                'name': eigen_results[2],
                'label': eigen_results[3],
                'color': eigen_results[4]
            }
        }
        
        # Create combined label
        combined_label = f"LBPH: {lbph_results[3]}\nEigen: {eigen_results[3]}"
        
        # Use green if both agree, orange if they disagree
        if lbph_results[2] == eigen_results[2] and lbph_results[2] != "Unknown":
            combined_color = (0, 255, 0)  # Green
        else:
            combined_color = (0, 165, 255)  # Orange
        
        return results, combined_label, combined_color
    
    else:
        # Unknown algorithm, default to LBPH
        return recognize_with_lbph(face_roi, clahe)


def format_comparison_text(results, font_scale=0.5):
    """
    Format comparison results for display
    
    Args:
        results: Results dictionary from recognize_face_unified with "Both"
        font_scale: Font scale for text sizing
        
    Returns:
        List of (text, color) tuples for each line
    """
    lines = []
    
    # LBPH line
    lbph = results['LBPH']
    lines.append((f"LBPH: {lbph['name']} ({lbph['confidence']:.0f})", lbph['color']))
    
    # Eigenfaces line
    eigen = results['Eigenfaces']
    lines.append((f"Eigen: {eigen['name']} ({eigen['confidence']:.0f})", eigen['color']))
    
    # Agreement indicator
    if lbph['name'] == eigen['name'] and lbph['name'] != "Unknown":
        lines.append(("✓ Match", (0, 255, 0)))
    elif lbph['name'] != eigen['name']:
        lines.append(("✗ Disagree", (0, 0, 255)))
    
    return lines
