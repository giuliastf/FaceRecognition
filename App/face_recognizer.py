import cv2
import numpy as np
import pickle
from recognition_unified import recognize_face_unified, format_comparison_text, get_algorithm

def recognize_faces():
    print("Starting face recognition...")
    
    # Check which algorithm is selected
    algorithm = get_algorithm()
    print(f"[INFO] Using algorithm: {algorithm}")
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    # Create CLAHE object for preprocessing (same as training)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    print("[INFO] Using CLAHE preprocessing for consistent recognition")

    font = cv2.FONT_HERSHEY_SIMPLEX
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)  # set video width
    cam.set(4, 480)  # set video height

    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)

    while True:
        ret, img = cam.read()
        if not ret:
            print("Failed to capture image")
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH)),
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Extract face region
            face_roi = gray[y:y + h, x:x + w]
            
            # Use unified recognition
            if algorithm == "Both":
                # Comparison mode - show both results
                results, combined_label, combined_color = recognize_face_unified(face_roi, clahe, algorithm)
                
                # Format text for display
                text_lines = format_comparison_text(results, font_scale=0.5)
                
                # Draw each line
                y_offset = y - 10
                for text, color in text_lines:
                    cv2.putText(img, text, (x + 5, y_offset), font, 0.5, color, 1)
                    y_offset -= 20
                    
            else:
                # Single algorithm mode
                id_, confidence, name, label, color = recognize_face_unified(face_roi, clahe, algorithm)
                
                cv2.putText(img, label, (x + 5, y - 5), font, 0.7, color, 2)
                
                # Add algorithm indicator
                algo_text = f"[{algorithm[:4]}]"
                cv2.putText(img, algo_text, (x + 5, y + h + 20), font, 0.5, (255, 255, 255), 1)

        # Add instruction text
        cv2.putText(img, "Press any key to exit", (10, 30), font, 0.7, (255, 255, 255), 2)
        
        # Add algorithm indicator at top
        algo_indicator = f"Algorithm: {algorithm}"
        cv2.putText(img, algo_indicator, (10, 60), font, 0.6, (0, 255, 255), 2)
        
        cv2.imshow('camera', img)
        k = cv2.waitKey(10) & 0xFF
        if k != 255:  # Any key pressed (255 means no key)
            break

    cam.release()
    cv2.destroyAllWindows()
    print("Face recognition finished.")
