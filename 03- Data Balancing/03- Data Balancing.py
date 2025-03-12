"""
get images from augmented folder balancing it to segmented folder 
ده الكود اللى احمد اشرف اللى عمله 


"""

import os
import numpy as np
from PIL import Image
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder
from collections import Counter

# Helper function to load images from a folder
def load_images_from_folder(folder_path, target_size=(128, 128)):
    images = []
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if file_path.endswith(('.png', '.jpg', '.jpeg')):
            try:
                img = Image.open(file_path).convert('RGB').resize(target_size)
                images.append(np.array(img))
            except Exception as e:
                print(f"Error processing file {file}: {e}")
    return images

# Paths to the folders
base_path = "D:/Learning/University of sadat/Grade 4/Semester 2/06- Graduation Project/Coding/Dogs Femur Fracture"
oblique_path = os.path.join(base_path, "Oblique/augmented")
overriding_path = os.path.join(base_path, "Overriding/augmented")

# Load images from both folders
oblique_images = load_images_from_folder(oblique_path)
overriding_images = load_images_from_folder(overriding_path)

# Number of images in both folders
num_oblique = len(oblique_images)
num_overriding = len(overriding_images)

# Combine the images and labels (0 for oblique, 1 for overriding)
X = np.array(oblique_images + overriding_images)
y = np.array([0] * num_oblique + [1] * num_overriding)

# Print the original class distribution
print("Original class distribution:", Counter(y))

# Apply Oversampling using RandomOverSampler to balance the classes
oversample = RandomOverSampler(sampling_strategy='minority', random_state=42)
X_resampled, y_resampled = oversample.fit_resample(X.reshape(X.shape[0], -1), y)

# Reshape the resampled images back to their original shape
X_resampled = X_resampled.reshape(X_resampled.shape[0], 128, 128, 3)

# Print the resampled class distribution
print("Resampled class distribution:", Counter(y_resampled))

# Create a target folder for the augmented synthetic images
augmented_folder = os.path.join(base_path, "Oblique/augmented_synthetic")
os.makedirs(augmented_folder, exist_ok=True)

# Save the resampled images to the new folder
for i, img in enumerate(X_resampled[:num_overriding]):  # We only save the same number as the Overriding class
    img_pil = Image.fromarray(img)
    img_pil.save(os.path.join(augmented_folder, f"augmented_{i+1}.png"))

print(f"Resampled and saved images to: {augmented_folder}")