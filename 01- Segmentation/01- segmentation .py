"""
get images from thier folders and segment it to segmented folder 
"""







""" 
################ The Real Code that segment the yellow box image ################
# import cv2
# import numpy as np
# import os

# # input_folder_2 = 'D:/Learning/University of sadat/Grade 4/Semester 1/06- Graduation Project/Coding/Dogs Femur Fracture/Oblique'
# # output_folder_2 = 'D:/Learning/University of sadat/Grade 4/Semester 1/06- Graduation Project/Coding/Dogs Femur Fracture/Oblique/augmented'
# # Paths to folders
# input_folder = 'D:/Learning/University of sadat/Grade 4/Semester 1/06- Graduation Project/Coding/Dogs Femur Fracture/Oblique'
# output_folder = 'D:/Learning/University of sadat/Grade 4/Semester 1/06- Graduation Project/Coding/Dogs Femur Fracture/Oblique/Segmented'
# os.makedirs(output_folder, exist_ok=True)

# # #### # Define the color of the bounding box (e.g., white box on black background)
# # #### # If the bounding box is a specific color, update the range
# # lower_color = (200, 200, 200)  # Lower bound of the box color (BGR)
# # upper_color = (255, 255, 255)  # Upper bound of the box color (BGR)
# # Define the HSV range for detecting yellow
# lower_yellow = np.array([20, 100, 100])  # Lower bound for yellow in HSV
# upper_yellow = np.array([30, 255, 255])  # Upper bound for yellow in HSV



# # Iterate through all images in the input folder
# for image_name in os.listdir(input_folder):
#     image_path = os.path.join(input_folder, image_name)

#     # Load the image
#     image = cv2.imread(image_path)
#     if image is None:
#         print(f"Failed to load {image_name}, skipping...")
#         continue

#     # Convert the image to HSV
#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

#     # Create a mask for yellow color
#     mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

#     # Find contours of the yellow bounding box
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # Process each detected contour
#     for contour in contours:
#         # Get bounding rectangle
#         x, y, w, h = cv2.boundingRect(contour)

#         # Optional: Filter small contours that might not be bounding boxes
#         if w * h < 500:  # Adjust the area threshold as needed
#             continue

#         # Crop the region of interest (ROI)
#         roi = image[y:y + h, x:x + w]

#         # Save the segmented image
#         output_path = os.path.join(output_folder, f"segmented_{image_name}")
#         cv2.imwrite(output_path, roi)
#         print(f"Processed and saved: {output_path}")

#         # Break if only one bounding box per image is expected
#         break



##############################################################################################################################


############################################### Removing the yellow box from the image by applying the black mask ########################################
# import cv2
# import numpy as np
# import os

# # Paths to folders
# input_folder = 'D:/Learning/University of sadat/Grade 4/Semester 1/06- Graduation Project/Coding/Dogs Femur Fracture/Oblique'
# output_folder = 'D:/Learning/University of sadat/Grade 4/Semester 1/06- Graduation Project/Coding/Dogs Femur Fracture/Oblique/Segmented'
# os.makedirs(output_folder, exist_ok=True)

# # Define the HSV range for detecting yellow
# lower_yellow = np.array([20, 100, 100])  # Lower bound for yellow in HSV
# upper_yellow = np.array([30, 255, 255])  # Upper bound for yellow in HSV

# # Iterate through all images in the input folder
# for image_name in os.listdir(input_folder):
#     image_path = os.path.join(input_folder, image_name)

#     # Load the image
#     image = cv2.imread(image_path)
#     if image is None:
#         print(f"Failed to load {image_name}, skipping...")
#         continue

#     # Convert the image to HSV
#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

#     # Create a mask for yellow color
#     mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

#     # Find contours of the yellow bounding box
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # Process each detected contour
#     for contour in contours:
#         # Get bounding rectangle
#         x, y, w, h = cv2.boundingRect(contour)

#         # Optional: Filter small contours that might not be bounding boxes
#         if w * h < 500:  # Adjust the area threshold as needed
#             continue

#         # Extract the region of interest (ROI) inside the yellow box
#         roi = image[y:y + h, x:x + w]

#         # Remove the yellow box by applying the mask
#         roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
#         yellow_mask = cv2.inRange(roi_hsv, lower_yellow, upper_yellow)
#         roi[yellow_mask > 0] = [0, 0, 0]  # Replace yellow with black

#         # Save the segmented ROI without the yellow box
#         output_path = os.path.join(output_folder, f"segmented_{image_name}")
#         cv2.imwrite(output_path, roi)
#         print(f"Processed and saved: {output_path}")

#         # Break if only one bounding box per image is expected
#         break
"""



############################### The Final Code  ###########################################
import cv2
import numpy as np
import os


def Segmentation(input_folder,output_folder):
    # Paths to folders
    input_folder = input_folder
    output_folder = output_folder

    # Define the HSV range for detecting yellow
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # Border width to remove (estimated thickness of the yellow line)
    border_thickness = 5  # Adjust based on your yellow box thickness

    # Iterate through all images in the input folder
    for image_name in os.listdir(input_folder):
        image_path = os.path.join(input_folder, image_name)

        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load {image_name}, skipping...")
            continue

        # Convert the image to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Create a mask for yellow color
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # Find contours of the yellow bounding box
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Process each detected contour
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)

            # Optional: Filter small contours that might not be bounding boxes
            if w * h < 500:  # Adjust the area threshold as needed
                continue

            # Crop the region of interest (ROI) while removing the yellow border
            cropped_roi = image[y + border_thickness:y + h - border_thickness,
                                x + border_thickness:x + w - border_thickness]

            # Save the cropped image
            output_path = os.path.join(output_folder, f"segmented_{image_name}")
            cv2.imwrite(output_path, cropped_roi)
            print(f"Processed and saved: {output_path}")

            # Break if only one bounding box per image is expected
            break

if __name__ == "__main__": 
    Base_Folder = 'D:/Learning/University of sadat/Grade 4/Semester 2/06- Graduation Project/Coding/' 
    #### # for oblique images
    Oblique_INPUT_FOLDER = f'{Base_Folder}00- Dogs Femur Fracture/Oblique/'
    Oblique_OUTPUT_FOLDER = f'{Base_Folder}Segmented_DataSet_Manuel/Oblique/'
    
    #### # for oblique images
    Overriding_INPUT_FOLDER = f'{Base_Folder}00- Dogs Femur Fracture/Overriding/'
    Overriding_OUTPUT_FOLDER = f'{Base_Folder}Segmented_DataSet_Manuel/Overriding/'
    
    os.makedirs(Oblique_OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(Overriding_OUTPUT_FOLDER, exist_ok=True)
    
    Segmentation(Oblique_INPUT_FOLDER,Oblique_OUTPUT_FOLDER)
    Segmentation(Overriding_INPUT_FOLDER,Overriding_OUTPUT_FOLDER)
