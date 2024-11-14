from PIL import Image
import os

def resize_and_crop_image(input_path, output_folder, remain_folder, target_height=512, crop_side="right"):
    with Image.open(input_path) as img:
        # Calculate the new width to maintain the aspect ratio
        aspect_ratio = img.width / img.height
        new_width = int(target_height * aspect_ratio)
        
        # Resize the image
        img_resized = img.resize((new_width, target_height), Image.Resampling.LANCZOS)

        # Determine crop parameters
        if crop_side == "right":
            cropped = img_resized.crop((new_width - 512, 0, new_width, target_height))
            remaining = img_resized.crop((0, 0, new_width - 512, target_height))
        elif crop_side == "left":
            cropped = img_resized.crop((0, 0, 512, target_height))
            remaining = img_resized.crop((512, 0, new_width, target_height))
        
        # Prepare output paths
        base_filename = os.path.basename(input_path)
        cropped_output_path = os.path.join(output_folder, base_filename)
        remaining_output_path = os.path.join(remain_folder, base_filename)
        
        # Save the cropped and remaining images
        cropped.save(cropped_output_path)
        remaining.save(remaining_output_path)

def process_images(input_folder, output_folder, remain_folder, target_height=512, crop_side="right"):
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        if os.path.isfile(input_path):
            resize_and_crop_image(input_path, output_folder, remain_folder, target_height, crop_side)

# Paths to the input folders
input_folder1 = '/home/B_UserData/sunleyao/UDIS2/video20240918/0_6/f'
input_folder2 = '/home/B_UserData/sunleyao/UDIS2/video20240918/0_6/b'

# Paths to the output folders
output_folder1 = '/home/B_UserData/sunleyao/UDIS2/video20240918/0_6/input1'
output_folder2 = '/home/B_UserData/sunleyao/UDIS2/video20240918/0_6/input2'
remain_output_folder1 = '/home/B_UserData/sunleyao/UDIS2/video20240918/0_6/remain_input1'
remain_output_folder2 = '/home/B_UserData/sunleyao/UDIS2/video20240918/0_6/remain_input2'

# Resize and crop images from both folders and save them to the respective output folders
process_images(input_folder1, output_folder1, remain_output_folder1, target_height=512, crop_side="right")
process_images(input_folder2, output_folder2, remain_output_folder2, target_height=512, crop_side="left")

print("Images have been resized, cropped, and saved successfully.")
