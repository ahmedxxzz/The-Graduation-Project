import random  
from shutil import copyfile 
import os 
# import sys 
# import zipfile 
# import shutil 
# from os import path, getcwd, chdir 
import tensorflow as tf 
# import keras 
from tensorflow.keras.optimizers import RMSprop   # type: ignore  # noqa: F401
# import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore



""" 
############# Data Augmentation ###################



# input_folder_1 = 'D:/Learning/University of sadat/Grade 4/Semester 1/06- Graduation Project/Coding/Dogs Femur Fracture/Overriding/Segmented'
# output_folder_1 = 'D:/Learning/University of sadat/Grade 4/Semester 1/06- Graduation Project/Coding/Dogs Femur Fracture/Overriding/augmented'

# input_folder_2 = 'D:/Learning/University of sadat/Grade 4/Semester 1/06- Graduation Project/Coding/Dogs Femur Fracture/Oblique/Segmented'
# output_folder_2 = 'D:/Learning/University of sadat/Grade 4/Semester 1/06- Graduation Project/Coding/Dogs Femur Fracture/Oblique/augmented'

# os.makedirs(output_folder_1, exist_ok=True)
# os.makedirs(output_folder_2, exist_ok=True)

# # Initialize the ImageDataGenerator for augmentation
# datagen = ImageDataGenerator(
#     rotation_range=40,          # Random rotation
#     width_shift_range=0.2,      # Horizontal shift
#     height_shift_range=0.2,     # Vertical shift
#     shear_range=0.2,            # Shearing
#     zoom_range=0.2,             # Zoom
#     horizontal_flip=True,       # Horizontal flip
#     fill_mode='nearest'         # Fill missing pixels
# )

# # Function to augment images from a folder and save to the corresponding output folder
# def augment_images(input_folder, output_folder):
#     # Get list of images in the input folder
#     images = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

#     for image_name in images:
#         img_path = os.path.join(input_folder, image_name)
#         img = Image.open(img_path)

#         # Check if the image has an alpha channel (RGBA) and convert it to RGB
#         if img.mode == 'RGBA':
#             img = img.convert('RGB')

#         img_array = tf.keras.preprocessing.image.img_to_array(img)
#         img_array = img_array.reshape((1,) + img_array.shape)

#         # Apply augmentation and save augmented images
#         for i, batch in enumerate(datagen.flow(img_array, batch_size=1, save_to_dir=output_folder, save_prefix='aug', save_format='jpeg')):
#             if i > 10:  # Number of augmented images per input image
#                 break

# # Augment images in both folders
# augment_images(input_folder_1, output_folder_1)
# augment_images(input_folder_2, output_folder_2)
###################### Augmentation After Edit ##########################
import random  
from shutil import copyfile 
import os 
import tensorflow as tf 
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
"""
############# Data Augmentation ###################
Base_Folder = 'D:/Learning/University of sadat/Grade 4/Semester 1/06- Graduation Project/Coding/'
input_folder_1 = f'{Base_Folder}Dogs Femur Fracture/Overriding/Segmented'
output_folder_1 = 'D:/Learning/University of sadat/Grade 4/Semester 1/06- Graduation Project/Coding/Dogs Femur Fracture/Overriding/augmented'

input_folder_2 = 'D:/Learning/University of sadat/Grade 4/Semester 1/06- Graduation Project/Coding/Dogs Femur Fracture/Oblique/Segmented'
output_folder_2 = 'D:/Learning/University of sadat/Grade 4/Semester 1/06- Graduation Project/Coding/Dogs Femur Fracture/Oblique/augmented'

os.makedirs(output_folder_1, exist_ok=True)
os.makedirs(output_folder_2, exist_ok=True)

# Initialize the ImageDataGenerator for augmentation
datagen = ImageDataGenerator(
    rotation_range=90,          
    horizontal_flip=True,       
    shear_range=0.2,            
    zoom_range=[0.8,1.2],             
    fill_mode='nearest'         
)

# Function to augment images from a folder and save to the corresponding output folder
def augment_images(input_folder, output_folder, num_images):
    # Get list of images in the input folder
    images = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for image_name in images:
        img_path = os.path.join(input_folder, image_name)
        img = Image.open(img_path)

        # Check if the image has an alpha channel (RGBA) and convert it to RGB
        if img.mode == 'RGBA':
            img = img.convert('RGB')

        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = img_array.reshape((1,) + img_array.shape)

        # Apply augmentation and save augmented images
        for i, batch in enumerate(datagen.flow(img_array, batch_size=1, save_to_dir=output_folder, save_prefix='aug', save_format='jpeg')):
            if i >= num_images - 1:  # Stop after generating the specified number of images
                break

# Specify the number of augmented images to generate per input image
num_augmented_images = 5

# Augment images in both folders
augment_images(input_folder_1, output_folder_1, num_augmented_images)
augment_images(input_folder_2, output_folder_2, num_augmented_images)


""" 
#################################################################################
# #################### Spliting Data into train and test ######################## 


# def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
#     files = []
#     # ## collect all the images
#     for filename in os.listdir(SOURCE):
#         file = SOURCE  +filename
#         if os.path.getsize(file) > 0:
#             files.append(filename)
#         else:
#             print(filename + " has not enough pixels to represent it as an image, seems corrupted so ignoring.")
#     print(files)
#     # ## Divide the images to training and testing 
#     training_length = int(len(files) * SPLIT_SIZE)
#     testing_length = int(len(files) - training_length)
#     shuffled_set = random.sample(files, len(files))
#     training_set = shuffled_set[0:training_length]
#     testing_set = shuffled_set[-testing_length:]

#     for filename in training_set:
#         this_file =  SOURCE  +filename
#         print(this_file)
#         destination = TRAINING  +filename
#         copyfile(this_file, destination)

#     for filename in testing_set:
#         this_file = SOURCE + filename
#         destination = TESTING + filename
#         copyfile(this_file, destination)




# Overriding_SOURCE_DIR = "D:/Learning/University of sadat/Grade 4/Semester 1/06- Graduation Project/Coding/Dogs Femur Fracture/overriding/augmented/"
# Oblique_SOURCE_DIR = "D:/Learning/University of sadat/Grade 4/Semester 1/06- Graduation Project/Coding/Dogs Femur Fracture/Oblique/augmented/"

# TRAINING_Overriding_DIR = "D:/Learning/University of sadat/Grade 4/Semester 1/06- Graduation Project/Coding/Data Augmented/training/overriding/"
# TESTING_Overriding_DIR = "D:/Learning/University of sadat/Grade 4/Semester 1/06- Graduation Project/Coding/Data Augmented/testing/overriding/"

# TRAINING_Oblique_DIR = "D:/Learning/University of sadat/Grade 4/Semester 1/06- Graduation Project/Coding/Data Augmented/training/Oblique/"
# TESTING_Oblique_DIR = "D:/Learning/University of sadat/Grade 4/Semester 1/06- Graduation Project/Coding/Data Augmented/testing/Oblique/"


# split_size = 0.9
# split_data(Overriding_SOURCE_DIR, TRAINING_Overriding_DIR, TESTING_Overriding_DIR, split_size)
# split_data(Oblique_SOURCE_DIR, TRAINING_Oblique_DIR, TESTING_Oblique_DIR, split_size)

# print("Total overriding images count :: ",len(os.listdir(TRAINING_Overriding_DIR)))
# print("Total Oblique images count :: ",len(os.listdir(TRAINING_Oblique_DIR)))

"""