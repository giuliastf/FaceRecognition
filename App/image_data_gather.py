import cv2
import os
import tkinter as tk
from tkinter import simpledialog, messagebox, filedialog
import time

global_face_id = None

def get_user_id():
    global global_face_id
    global_face_id = simpledialog.askstring("Input", "Who is this?")
    return global_face_id

def collect_images():
    print("[DEBUG] collect_images() function called!")
    global global_face_id
    face_id = global_face_id
    print(f"[DEBUG] face_id = {face_id}")
    if not face_id:
        messagebox.showwarning("Error", "No user id entered. Please try again.")
        return

    print(f"[DEBUG] Starting image collection for: {face_id}")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[DEBUG] Camera 0 failed, trying camera 1...")
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open camera. Check permissions.")
            return
    
    print("[DEBUG] Camera opened successfully")

    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if face_detector.empty():
        messagebox.showerror("Error", "Could not load Haar cascade")
        return
    
    print("[DEBUG] Face detector loaded")

    dataset_path = os.path.join('dataset', face_id)
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
        print(f"[INFO] Created directory: {dataset_path}")
    else:
        print(f"[INFO] Directory already exists: {dataset_path}")

    count = 0
    start_time = time.time()
    
    print("[DEBUG] Starting camera loop...")
    
    # Test if we can read a frame first
    ret, test_frame = cap.read()
    if not ret:
        messagebox.showerror("Error", "Cannot read from camera")
        cap.release()
        return
    
    print(f"[DEBUG] Camera frame size: {test_frame.shape}")
    
    # Create named window for proper display
    cv2.namedWindow("Collecting faces - press ESC to stop", cv2.WINDOW_NORMAL)
    
    # Get reference to tkinter root for updates
    import tkinter as tk
    root = tk._default_root
    
    frames_processed = 0
    
    while True:
        ret, img = cap.read()
        if not ret:
            print("Error: Could not read frame from video capture")
            break
        
        frames_processed += 1
        if frames_processed % 10 == 0:
            print(f"[DEBUG] Processed {frames_processed} frames, saved {count} faces")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
        
        if len(faces) > 0:
            print(f"[DEBUG] Detected {len(faces)} face(s)")

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            count += 1
            image_path = os.path.join(dataset_path, f"{str(count)}.jpg")
            cv2.imwrite(image_path, gray[y:y + h, x:x + w])
            print(f"[INFO] Saved image: {image_path}")
            
            # Add text overlay showing progress
            cv2.putText(img, f"Captured: {count}/30", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the frame with rectangles and progress
        try:
            cv2.imshow("Collecting faces - press ESC to stop", img)
        except Exception as e:
            print(f"[DEBUG] Display error: {e}")
            # Continue without display if needed

        # Handle key input and exit conditions
        k = cv2.waitKey(100) & 0xff
        if k == 27 or count >= 30:  # ESC key or 30 images
            print(f"[DEBUG] Stopping: ESC pressed or count reached ({count})")
            break
            
        # Keep tkinter responsive (CRITICAL for main thread integration)
        if root:
            try:
                root.update()
            except Exception as e:
                print(f"[DEBUG] Tkinter update error: {e}")
            
        # Safety timeout (60 seconds)
        if time.time() - start_time > 60:
            print("[INFO] Timeout reached")
            break

    print(f"[DEBUG] Collection finished. Total images: {count}")
    cap.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("Info", f"Collected {count} images for user {face_id}")

def upload_training_images():
    """Upload multiple images and process them like camera collection"""
    # Get person name
    face_id = simpledialog.askstring("Input", "Who is this person? (for uploaded images)")
    if not face_id:
        messagebox.showwarning("Error", "No person name entered.")
        return
    
    print(f"[DEBUG] Starting image upload for: {face_id}")
    
    # Select multiple image files
    file_paths = filedialog.askopenfilenames(
        title="Select training images for " + face_id,
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("PNG files", "*.png"),
            ("All files", "*.*")
        ]
    )
    
    if not file_paths:
        messagebox.showinfo("Cancelled", "No images selected.")
        return
    
    print(f"[DEBUG] Selected {len(file_paths)} images")
    
    # Initialize face detector
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if face_detector.empty():
        messagebox.showerror("Error", "Could not load Haar cascade")
        return
    
    # Create dataset directory
    dataset_path = os.path.join('dataset', face_id)
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
        print(f"[INFO] Created directory: {dataset_path}")
    else:
        print(f"[INFO] Directory already exists: {dataset_path}")
    
    # Check existing files to continue numbering
    existing_files = [f for f in os.listdir(dataset_path) if f.endswith('.jpg')]
    if existing_files:
        # Find the highest number and start from there
        numbers = []
        for f in existing_files:
            try:
                num = int(f.split('.')[0])
                numbers.append(num)
            except ValueError:
                continue
        count = max(numbers) if numbers else 0
        print(f"[INFO] Found {len(existing_files)} existing images. Starting from number {count + 1}")
    else:
        count = 0
        print(f"[INFO] No existing images found. Starting from number 1")
    processed_images = 0
    faces_found = 0
    
    for file_path in file_paths:
        processed_images += 1
        print(f"[INFO] Processing image {processed_images}/{len(file_paths)}: {os.path.basename(file_path)}")
        
        # Load image
        img = cv2.imread(file_path)
        if img is None:
            print(f"[WARNING] Could not load image: {file_path}")
            continue
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces with improved parameters
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
        
        if len(faces) == 0:
            print(f"[WARNING] No faces detected in: {os.path.basename(file_path)}")
            continue
        
        faces_found += len(faces)
        
        # Process each detected face (same as camera collection)
        for (x, y, w, h) in faces:
            count += 1
            face_crop = gray[y:y + h, x:x + w]  # Extract face region
            
            # Save cropped face (same format as camera collection)
            save_path = os.path.join(dataset_path, f"{count}.jpg")
            cv2.imwrite(save_path, face_crop)
            print(f"[INFO] Saved face: {save_path}")
    
    # Results summary
    message = f"""Upload completed!

Person: {face_id}
Images processed: {processed_images}
Faces detected: {faces_found}
Face images saved: {count}

Ready for training!"""
    
    messagebox.showinfo("Upload Complete", message)
    print(f"[DEBUG] Upload finished. Total face images saved: {count}")
