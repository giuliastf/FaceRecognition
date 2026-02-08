import tkinter as tk
from tkinter import ttk
from threading import Thread
from image_upload import upload_and_recognize_image
from image_data_gather import collect_images, get_user_id, global_face_id, upload_training_images
from face_training import train_model
from face_recognizer import recognize_faces
from recognition_unified import set_algorithm, get_algorithm
import os

# Global variable for algorithm selection
algorithm_var = None

def show_eigenfaces_visualization():
    """Show eigenfaces visualization window"""
    from face_training_eigen import load_eigenfaces_model
    import cv2
    import numpy as np
    from tkinter import messagebox
    
    model_path = 'trainer/trainer_eigen.pkl'
    if not os.path.exists(model_path):
        messagebox.showerror("Error", "No Eigenfaces model found. Please train the model first.")
        return
    
    model = load_eigenfaces_model(model_path)
    if model is None or not model.person_models:
        messagebox.showerror("Error", "Failed to load Eigenfaces model.")
        return
    
    # Create visualization
    print("[INFO] Generating Eigenfaces visualization...")
    
    for person_id in sorted(model.person_models.keys()):
        person_name = model.person_models[person_id]['person_name']
        mean_face, eigenfaces = model.get_eigenfaces(person_id, max_components=5)
        
        if mean_face is None:
            continue
        
        # Create combined image
        num_faces = len(eigenfaces) + 1
        combined_width = mean_face.shape[1] * num_faces
        combined_height = mean_face.shape[0]
        combined_img = np.zeros((combined_height, combined_width))
        
        # Add mean face
        combined_img[:, 0:mean_face.shape[1]] = mean_face
        
        # Add eigenfaces
        for i, eigenface in enumerate(eigenfaces):
            x_offset = (i + 1) * mean_face.shape[1]
            combined_img[:, x_offset:x_offset + eigenface.shape[1]] = eigenface
        
        # Convert to uint8 and add labels
        combined_img = (combined_img * 255).astype(np.uint8)
        combined_img_color = cv2.cvtColor(combined_img, cv2.COLOR_GRAY2BGR)
        
        # Add title
        cv2.putText(combined_img_color, f"{person_name} - Mean Face + Eigenfaces", 
                   (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show window
        cv2.imshow(f'Eigenfaces - {person_name}', combined_img_color)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    messagebox.showinfo("Info", "Eigenfaces visualization complete.")

def create_main_ui():
    global algorithm_var
    
    root = tk.Tk()
    
    root.title("Face Recognition System - Multi-Algorithm")
    root.geometry("550x650")
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
    
    # Algorithm selection frame
    algo_frame = tk.LabelFrame(main_frame, text="Recognition Algorithm", 
                               font=('Helvetica', 11, 'bold'), 
                               bg='#34e5eb', fg='#2c3e50', 
                               padx=10, pady=10)
    algo_frame.pack(pady=10, fill='x')
    
    algorithm_var = tk.StringVar(value="LBPH")
    
    def on_algorithm_change():
        selected = algorithm_var.get()
        set_algorithm(selected)
        print(f"[GUI] Algorithm changed to: {selected}")
    
    # Radio buttons for algorithm selection
    radio_style = {'bg': '#34e5eb', 'fg': '#2c3e50', 'font': ('Helvetica', 10),
                   'activebackground': '#34e5eb', 'selectcolor': '#6a1fcc'}
    
    lbph_radio = tk.Radiobutton(algo_frame, text="LBPH (Local Binary Patterns) - Fast, Light-Robust", 
                                variable=algorithm_var, value="LBPH", 
                                command=on_algorithm_change, **radio_style)
    lbph_radio.pack(anchor='w', pady=2)
    
    eigen_radio = tk.Radiobutton(algo_frame, text="Eigenfaces (PCA) - Global Shape-Based", 
                                 variable=algorithm_var, value="Eigenfaces", 
                                 command=on_algorithm_change, **radio_style)
    eigen_radio.pack(anchor='w', pady=2)
    
    both_radio = tk.Radiobutton(algo_frame, text="Both (Comparison Mode) - Side-by-Side", 
                                variable=algorithm_var, value="Both", 
                                command=on_algorithm_change, **radio_style)
    both_radio.pack(anchor='w', pady=2)

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

    train_button = tk.Button(main_frame, text="Train Models (LBPH + Eigenfaces)", command=lambda: Thread(target=train_model).start(), **button_style)
    train_button.pack(pady=8)

    recognize_button = tk.Button(main_frame, text="Recognize Faces (Camera)", command=recognize_faces, **button_style)
    recognize_button.pack(pady=8)

    upload_button = tk.Button(main_frame, text="Upload and Scan Image", command=upload_and_recognize_image, **button_style)
    upload_button.pack(pady=8)
    
    # New button for Eigenfaces visualization
    eigenfaces_button = tk.Button(main_frame, text="Show Eigenfaces (Ghost Faces)", 
                                  command=show_eigenfaces_visualization, **button_style)
    eigenfaces_button.pack(pady=8)

    root.mainloop()

if __name__ == "__main__":
    create_main_ui()
