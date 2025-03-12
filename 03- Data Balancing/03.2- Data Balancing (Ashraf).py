import os
import random
import numpy as np
from PIL import Image, ImageEnhance
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

new_folder = 'dataset (1)/balanced_Oblique'
if not os.path.exists(new_folder):
    os.makedirs(new_folder)

folder_oblique = 'dataset (1)/Oblique'
folder_overriding = 'dataset (1)/Overriding'

image_size = (224, 224)

datagen = ImageDataGenerator(
    horizontal_flip=True,
    rotation_range=30,
    brightness_range=(0.8, 1.2),
    zoom_range=0.2,
    vertical_flip=True
)

def load_images_from_folder(folder_path, label, image_size=image_size):
    images = []
    labels = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path).convert("RGB").resize(image_size)
            images.append(np.array(img))
            labels.append(label)
    return images, labels

images_oblique, labels_oblique = load_images_from_folder(folder_oblique, label=0)
images_overriding, labels_overriding = load_images_from_folder(folder_overriding, label=1)

oblique_images_count = len(images_oblique)
overriding_images_count = len(images_overriding)

final_count = 1173
num_augmented_images_needed = final_count - oblique_images_count
print(f"Number of augmented images needed: {num_augmented_images_needed}")

augmented_images = []
augmented_labels = []

for i in range(num_augmented_images_needed):
    img = random.choice(images_oblique)
    img = np.expand_dims(img, axis=0)  
    augmented_img = datagen.flow(img, batch_size=1)[0].astype(np.uint8)[0]
    augmented_images.append(augmented_img)
    augmented_labels.append(labels_oblique[0])

final_images = images_oblique + augmented_images
final_labels = labels_oblique + augmented_labels

def save_images_to_folder(images, folder_path):
    for idx, img_array in enumerate(images):
        img = Image.fromarray(img_array)
        img_name = f"generated_{idx}.jpeg"
        img.save(os.path.join(folder_path, img_name))

save_images_to_folder(final_images, new_folder)

print(f"Saved {len(final_images)} images in the folder {new_folder}.")

