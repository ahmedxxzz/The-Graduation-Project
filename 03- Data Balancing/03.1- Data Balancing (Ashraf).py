import os
import random
from PIL import Image
import torch
from torchvision import transforms as T

new_folder = 'dataset (1)/balanced_Oblique'

if not os.path.exists(new_folder):
    os.makedirs(new_folder)

folder_oblique = 'dataset (1)/Oblique'
folder_overriding = 'dataset (1)/Overriding'

image_size = (128, 128)

transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomRotation(30),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    T.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
    T.RandomVerticalFlip(),
    T.ToTensor(),
])

def load_images_from_folder(folder_path, label, image_size=image_size):
    images = []
    labels = []
    for filename in os.listdir(folder_path):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path).convert("RGB")
            img = img.resize(image_size)
            images.append(img)
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
    augmented_img = transform(img)
    augmented_images.append(augmented_img)
    augmented_labels.append(labels_oblique[0])

final_images = images_oblique + augmented_images
final_labels = labels_oblique + augmented_labels

def save_images_to_folder(images, labels, folder_path):
    for idx, img_tensor in enumerate(images):
        img_name = f"generated_{idx}.jpeg"
        if isinstance(img_tensor, torch.Tensor):
            img = T.ToPILImage()(img_tensor.cpu())
        else:
            img = img_tensor
        img.save(os.path.join(folder_path, img_name))

save_images_to_folder(final_images, final_labels, new_folder)

print(f"Saved {len(final_images)} images in the folder {new_folder}.")

