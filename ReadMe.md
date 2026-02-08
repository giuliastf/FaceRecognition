# Face Recognition System

<img align="right" src="Resources/icon.png" width="25%">

A comprehensive face recognition application built with Python, OpenCV, and Tkinter that can collect training data, train a recognition model, and perform real-time face identification.

---

## Table of Contents
- [Overview](#overview)
- [How It Works](#how-it-works)
  - [System Architecture](#system-architecture)
  - [Face Detection](#face-detection)
  - [Image Preprocessing](#image-preprocessing)
  - [Recognition Algorithm](#recognition-algorithm)
- [Features](#features)
- [Workflow](#workflow)
- [Technical Implementation](#technical-implementation)
- [Project Structure](#project-structure)
- [Usage Guide](#usage-guide)
- [Documentation](#documentation)

---

## Overview

This Face Recognition System enables users to:
1. **Collect training images** (via webcam or file upload)
2. **Train a recognition model** using collected data
3. **Recognize faces** in real-time via webcam
4. **Analyze uploaded images** for face identification

The system uses **Haar Cascade** for face detection and **LBPH (Local Binary Patterns Histograms)** for face recognition, making it efficient and accurate for real-world applications.

---

## How It Works

### System Architecture

The application follows a machine learning pipeline:
```
Data Collection → Preprocessing → Model Training → Recognition → Results
```

### Face Detection

**Haar Cascade Classifier** is used to detect faces in images:
- Uses machine learning-based approach with cascaded classifiers
- Focuses on key facial features: **eyes, nose, and face contours**
- Detects facial regions by analyzing:
  - **Dark regions** (eyes, eyebrows)
  - **Light regions** (nose bridge, cheeks)
  - **Edge patterns** (face boundaries)

<p align="center">
  <img src="./Resources/Features.jpg" alt="Features on human faces">
</p>

### Image Preprocessing

To ensure consistent and accurate recognition, images undergo several preprocessing steps:

1. **Grayscale Conversion**: RGB images → grayscale (reduces complexity)
2. **CLAHE Enhancement** (Contrast Limited Adaptive Histogram Equalization):
   - Normalizes lighting conditions
   - Enhances local contrast
   - Parameters: `clipLimit=2.0`, `tileGridSize=(8,8)`
3. **Face Resizing**: All faces resized to 150×150 pixels for consistency
4. **Face Region Extraction**: Only the detected face region is used

### Recognition Algorithm

**LBPH (Local Binary Patterns Histograms)** Face Recognizer:

1. **Training Phase**:
   - Analyzes texture patterns in each face image
   - Creates a histogram representation of local binary patterns
   - Assigns a unique ID to each person
   - Stores trained model in `trainer/trainer.yml`
   - Saves label mappings (ID ↔ Name) in `trainer/labels.pickle`

2. **Recognition Phase**:
   - Extracts LBP features from detected face
   - Compares with trained model
   - Returns: 
     - **ID**: Person identifier
     - **Confidence**: Lower = better match (0-100)
   
3. **Confidence Thresholds**:
   - **< 80**: High confidence (Green) ✓
   - **80-100**: Medium confidence (Orange) ?
   - **> 100**: Low confidence - Unknown (Red) ✗

---

## Features

### 1. **Camera Data Collection**
- Captures 30 training images via webcam
- Real-time face detection with bounding boxes
- Progress indicator showing captured images (e.g., "Captured: 15/30")
- Automatic saving to `dataset/[PersonName]/` folder
- Press ESC to stop early

### 2. **Image Upload for Training**
- Upload multiple existing photos at once
- Supports formats: JPG, JPEG, PNG, BMP, TIFF
- Automatically detects and crops faces from images
- Allows batch processing of training data

### 3. **Model Training**
- Trains LBPH recognizer on collected dataset
- Applies CLAHE preprocessing for lighting normalization
- Incremental training: can add new people without retraining from scratch
- Saves trained model and label mappings
- Progress feedback via message boxes

### 4. **Real-Time Face Recognition**
- Live camera feed with face detection
- Real-time name prediction with confidence scores
- Color-coded results:
  - **Green**: High confidence match
  - **Orange**: Medium confidence (uncertain)
  - **Red**: Unknown person
- Press any key to exit

### 5. **Image Upload Recognition**
- Upload any image to detect and recognize faces
- **Adaptive text scaling** based on image size
- Multiple face detection in single image
- Visual feedback with bounding boxes and labels
- Close with any key or X button

### 6. **User-Friendly Interface**
- Clean Tkinter GUI with custom styling
- Color scheme: Cyan background (`#34e5eb`) with purple buttons (`#6a1fcc`)
- Clear button labels and intuitive workflow
- Threading for non-blocking operations

---

## Workflow

### Step-by-Step Process

```
┌─────────────────────────────────────────────────────┐
│  1. COLLECT TRAINING DATA                           │
│     ├─ Camera: Capture 30 photos                    │
│     └─ Upload: Select existing images               │
└─────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────┐
│  2. TRAIN MODEL                                     │
│     ├─ Load all images from dataset/               │
│     ├─ Detect faces with Haar Cascade              │
│     ├─ Preprocess: Grayscale + CLAHE + Resize      │
│     ├─ Train LBPH recognizer                        │
│     └─ Save model (trainer.yml) & labels            │
└─────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────┐
│  3. RECOGNITION                                     │
│     ├─ Camera: Real-time recognition                │
│     └─ Upload: Analyze static images                │
└─────────────────────────────────────────────────────┘
```

### Example Workflow

1. **Add Person "John"**:
   - Click "Collect Images (Camera)"
   - Enter "John" when prompted
   - System captures 30 photos → saved to `dataset/John/`

2. **Add Person "Jane"** (from existing photos):
   - Click "Upload Training Images"
   - Select Jane's photos
   - System processes and saves to `dataset/Jane/`

3. **Train the Model**:
   - Click "Train Model"
   - System processes all images in `dataset/`
   - Model saved to `trainer/trainer.yml`

4. **Recognize Faces**:
   - Click "Recognize Faces" for live camera
   - OR click "Upload and Scan Image" for static image analysis
   - System identifies John/Jane with confidence scores

---

## Technical Implementation

### Key Technologies

- **Python 3.x**
- **OpenCV (cv2)**: Computer vision library for face detection/recognition
- **Tkinter**: GUI framework
- **NumPy**: Numerical operations
- **Pickle**: Model/label serialization
- **PIL (Pillow)**: Image processing
- **Threading**: Asynchronous operations

### Core Components

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| `main.py` | Main GUI and application entry point | `create_main_ui()` |
| `camera_data_gather.py` | Webcam image collection | `collect_images()` |
| `image_data_gather.py` | File upload and processing | `upload_training_images()` |
| `face_training.py` | Model training with LBPH | `train_model()` |
| `face_recognizer.py` | Real-time camera recognition | `recognize_faces()` |
| `image_upload.py` | Static image analysis | `upload_and_recognize_image()` |

### File Structure

```
FaceRecognition/
├── App/
│   ├── main.py                    # Main GUI application
│   ├── camera_data_gather.py      # Webcam data collection
│   ├── image_data_gather.py       # Image upload & processing
│   ├── face_training.py           # Model training logic
│   ├── face_recognizer.py         # Real-time recognition
│   ├── image_upload.py            # Static image recognition
│   └── style.py                   # UI styling
├── dataset/                       # Training images (organized by person)
│   ├── Person1/
│   │   ├── 1.jpg
│   │   ├── 2.jpg
│   │   └── ...
│   └── Person2/
│       └── ...
├── trainer/
│   ├── trainer.yml                # Trained LBPH model
│   └── labels.pickle              # ID ↔ Name mappings
├── Resources/                     # Icons and documentation
└── README.md                      # This file
```

---

## Usage Guide

### Installation Requirements

```bash
pip install opencv-python opencv-contrib-python numpy pillow
```

### Running the Application

```bash
cd App
python main.py
```

### GUI Buttons

| Button | Action |
|--------|--------|
| **Collect Images (Camera)** | Opens webcam to capture 30 training photos |
| **Upload Training Images** | Select existing photos from your computer |
| **Train Model** | Trains the recognition model on collected data |
| **Recognize Faces** | Opens camera for real-time face recognition |
| **Upload and Scan Image** | Analyze and recognize faces in uploaded image |

### Tips for Best Results

✓ **Training Data**:
- Collect at least 30 images per person
- Use varied angles and expressions
- Ensure good lighting conditions
- Face should be clearly visible

✓ **Recognition**:
- Maintain similar lighting to training photos
- Face the camera directly
- Avoid obstructions (sunglasses, masks)

---

## Recent Improvements

### 1. Adaptive Text Scaling
- Text size automatically adjusts based on image dimensions
- Font scale: 0.4 to 1.2 (dynamic)
- Rectangle thickness scales with face size
- Black background for text readability

### 2. Improved Window Controls
- **Image Recognition**: Press any key or click X to close
- **Camera Recognition**: Press any key to exit
- Visual instructions displayed on screen
- Responsive window handling (100ms polling)

### 3. macOS Compatibility
- Fixed OpenCV GUI threading issue
- Recognition runs on main thread (required for macOS)
- Training runs in background thread (non-blocking)

### 4. User Experience Enhancements
- Real-time progress indicators
- Color-coded confidence levels
- Clear on-screen instructions
- Smooth workflow between features

---

## Documentation

- **[Project Report](./Resources/AI_LAB_Project_Report.pdf)** - Detailed technical documentation

---

## Algorithm Details

### LBPH Face Recognition Process

1. **Divide face into cells** (e.g., 8×8 grid)
2. **For each cell**:
   - Compare each pixel with neighbors (8 surrounding pixels)
   - Create binary pattern (1 if neighbor > center, 0 otherwise)
   - Convert binary to decimal (0-255)
3. **Create histogram** of patterns for each cell
4. **Concatenate histograms** → final feature vector
5. **Compare feature vectors** using Chi-square distance

**Advantages**:
- ✓ Robust to illumination changes
- ✓ Fast computation
- ✓ Works with grayscale images
- ✓ Simple to implement

---

## System Requirements

- **OS**: Windows, macOS, Linux
- **Python**: 3.7+
- **Webcam**: For data collection and real-time recognition
- **Storage**: Minimal (models ~1MB per person)

---

## Future Enhancements

- [ ] Add support for multiple face recognition models
- [ ] Implement face verification (1:1 matching)
- [ ] Add age/gender detection
- [ ] Export recognition logs
- [ ] Mobile app integration

---

## Contributors

Developed as part of an AI Lab project demonstrating practical applications of computer vision and machine learning.

---

## License

This project is for educational purposes. Please ensure compliance with privacy regulations when using face recognition technology.

