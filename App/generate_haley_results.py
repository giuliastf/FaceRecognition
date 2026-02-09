"""
Generate Algorithm Results for Haley Dunphy
- Eigenfaces: Mean face + Principal Components (PCA result)
- LBPH: LBP patterns and histogram extraction (LBPH process)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from face_training_eigen import load_eigenfaces_model


def compute_lbp_image(image):
    """Compute LBP pattern for visualization"""
    height, width = image.shape
    lbp_image = np.zeros((height-2, width-2), dtype=np.uint8)
    
    for i in range(1, height-1):
        for j in range(1, width-1):
            center = image[i, j]
            neighbors = [
                image[i-1, j-1], image[i-1, j], image[i-1, j+1],
                image[i, j+1], image[i+1, j+1], image[i+1, j],
                image[i+1, j-1], image[i, j-1]
            ]
            binary_string = ''.join(['1' if n >= center else '0' for n in neighbors])
            lbp_value = int(binary_string, 2)
            lbp_image[i-1, j-1] = lbp_value
    
    return lbp_image


def extract_histogram(lbp_image):
    """Extract histogram for visualization"""
    hist = cv2.calcHist([lbp_image], [0], None, [256], [0, 256])
    return hist.flatten()


def generate_haley_eigenfaces():
    """Generate Eigenfaces visualization for Haley"""
    print("\n[1/2] Generating Eigenfaces (PCA) Result for Haley...")
    
    model_path = '../trainer/trainer_eigen.pkl'
    if not os.path.exists(model_path):
        model_path = 'trainer/trainer_eigen.pkl'
    
    if not os.path.exists(model_path):
        print("[ERROR] Eigenfaces model not found. Train the model first.")
        return None
    
    model = load_eigenfaces_model(model_path)
    if model is None or not model.person_models:
        print("[ERROR] Failed to load Eigenfaces model.")
        return None
    
    # Find Haley's ID
    haley_id = None
    for person_id, data in model.person_models.items():
        if 'Haley' in data['person_name']:
            haley_id = person_id
            break
    
    if haley_id is None:
        print("[ERROR] Haley not found in Eigenfaces model.")
        return None
    
    # Get eigenfaces
    mean_face, eigenfaces = model.get_eigenfaces(haley_id, max_components=5)
    
    if mean_face is None:
        print("[ERROR] Could not get eigenfaces for Haley.")
        return None
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Eigenfaces (PCA) Result for Haley Dunphy', 
                 fontsize=16, fontweight='bold')
    
    # Mean face (large)
    axes[0, 0].imshow(mean_face, cmap='gray')
    axes[0, 0].set_title('Mean Face\n(Average of all Haley images)', 
                         fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Eigenfaces (principal components)
    for i, eigenface in enumerate(eigenfaces):
        row = (i + 1) // 3
        col = (i + 1) % 3
        axes[row, col].imshow(eigenface, cmap='gray')
        axes[row, col].set_title(f'Eigenface {i+1}\n(Principal Component {i+1})', 
                                fontsize=11, fontweight='bold')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    # Save
    output_dir = '../output_images'
    if not os.path.exists(output_dir):
        output_dir = 'output_images'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    output_path = os.path.join(output_dir, 'haley_eigenfaces_result.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_path}")
    return output_path


def generate_haley_lbph():
    """Generate LBPH process visualization for Haley"""
    print("\n[2/2] Generating LBPH Result for Haley...")
    
    # Find Haley's images
    dataset_path = '../dataset/Haley Dunphy'
    if not os.path.exists(dataset_path):
        dataset_path = 'dataset/Haley Dunphy'
    
    if not os.path.exists(dataset_path):
        print("[ERROR] Haley Dunphy folder not found in dataset")
        return None
    
    # Get first image
    images = [f for f in os.listdir(dataset_path) if f.endswith('.jpg')]
    if not images:
        print("[ERROR] No images found for Haley")
        return None
    
    img_path = os.path.join(dataset_path, images[0])
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    enhanced = cv2.resize(enhanced, (150, 150))
    
    # Compute LBP
    lbp_image = compute_lbp_image(enhanced)
    
    # Extract histogram
    hist = extract_histogram(lbp_image)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('LBPH Result for Haley Dunphy', fontsize=16, fontweight='bold')
    
    # Original preprocessed
    axes[0].imshow(enhanced, cmap='gray')
    axes[0].set_title('1. Preprocessed Face\n(Grayscale + CLAHE)', 
                        fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # LBP pattern
    axes[1].imshow(lbp_image, cmap='hot')
    axes[1].set_title('2. LBP Pattern\n(Texture Map)', 
                        fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Histogram
    axes[2].bar(range(256), hist, width=1.0, color='blue', alpha=0.7)
    axes[2].set_title('3. LBP Histogram\n(Feature Vector)', 
                        fontsize=12, fontweight='bold')
    axes[2].set_xlabel('LBP Value (0-255)', fontsize=10)
    axes[2].set_ylabel('Frequency', fontsize=10)
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_dir = '../output_images'
    if not os.path.exists(output_dir):
        output_dir = 'output_images'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    output_path = os.path.join(output_dir, 'haley_lbph_result.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_path}")
    return output_path


def main():
    print("="*60)
    print("GENERATING ALGORITHM RESULTS FOR HALEY DUNPHY")
    print("="*60)
    
    eigen_path = generate_haley_eigenfaces()
    lbph_path = generate_haley_lbph()
    
    print("\n" + "="*60)
    print("✓ DONE!")
    print("="*60)
    
    if eigen_path:
        print(f"\nEigenfaces Result: {eigen_path}")
        print("  → Shows mean face + 5 principal components (PCA)")
    
    if lbph_path:
        print(f"\nLBPH Result: {lbph_path}")
        print("  → Shows LBP pattern + histogram (no 'face' images)")
    
    print("\n💡 Use these images in your report to show:")
    print("   • What Eigenfaces (PCA) produces: visual face images")
    print("   • What LBPH produces: texture patterns & histograms")


if __name__ == "__main__":
    main()
