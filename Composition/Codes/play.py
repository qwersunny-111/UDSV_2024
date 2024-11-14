import cv2
import os

def images_to_video(input_folder, output_video, frame_rate=30):
    # Get the list of image files
    images = [img for img in os.listdir(input_folder) if img.endswith(".jpg")]

    # Sort the images by filename
    images.sort(key=lambda x: int(x.split('.')[0]))

    # Determine the width and height of the images
    img = cv2.imread(os.path.join(input_folder, images[0]))
    height, width, _ = img.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height))

    # Write images to video
    for image in images:
        img_path = os.path.join(input_folder, image)
        frame = cv2.imread(img_path)
        out.write(frame)

    # Release VideoWriter object
    out.release()

# Paths to the input folder and output video
input_folder = '/home/sunleyao/sly/UDIS2-main/UDIS2++-control points/Composition/composition'
output_video = '/home/sunleyao/sly/UDIS2-main/UDIS2++-control points/Composition/composition/output_video.avi'

# Convert images to video
images_to_video(input_folder, output_video)

print("Video has been created successfully.")
