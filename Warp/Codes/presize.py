from PIL import Image
import os

def resize_image(input_path, output_path, target_height=512):
    # Open the image
    with Image.open(input_path) as img:
        # Calculate the new width to maintain the aspect ratio
        aspect_ratio = img.width / img.height
        new_width = int(target_height * aspect_ratio)
        
        # Resize the image
        img_resized = img.resize((new_width, target_height), Image.Resampling.LANCZOS)
        
        # Save the resized image to the specified path
        img_resized.save(output_path)

def process_images(input_folder1, input_folder2, output_folder1, output_folder2, target_height=512):
    # Process images from the first input folder
    for filename in os.listdir(input_folder1):
        input_path = os.path.join(input_folder1, filename)
        if os.path.isfile(input_path):
            output_path = os.path.join(output_folder1, filename)
            resize_image(input_path, output_path, target_height)
    
    # Process images from the second input folder
    for filename in os.listdir(input_folder2):
        input_path = os.path.join(input_folder2, filename)
        if os.path.isfile(input_path):
            output_path = os.path.join(output_folder2, filename)
            resize_image(input_path, output_path, target_height)

# Paths to the input folders
input_folder1 = '/home/B_UserData/sunleyao/UDIS2/video20240918/1_9/f'
input_folder2 = '/home/B_UserData/sunleyao/UDIS2/video20240918/1_9/b'

# Paths to the output folders
output_folder1 = '/home/B_UserData/sunleyao/UDIS2/video20240918/1_9/resize_f'
output_folder2 = '/home/B_UserData/sunleyao/UDIS2/video20240918/1_9/resize_b'

# Resize images from both folders and save them to the respective output folders
process_images(input_folder1, input_folder2, output_folder1, output_folder2)

print("Images have been resized and saved successfully.")
