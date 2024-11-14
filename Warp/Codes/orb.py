import cv2
import numpy as np
import time

def orb(image):
    start_time = time.time()
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Initialize the ORB detector
    orb = cv2.ORB_create()
    
    # Find the keypoints and descriptors with ORB
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"execution time: {execution_time:.2f} seconds")
    # Draw keypoints on the image with a unified color
    orb_image = cv2.drawKeypoints(image, keypoints, None, color=(0, 225, 0), flags=0)
    
    return orb_image

# Read the image
image_path = '/home/B_UserData/sunleyao/lyc/0010/0.jpg'
image = cv2.imread(image_path)


# Extract ORB features
orb_image = orb(image)

# Save the image to a specified path
output_path = '/home/B_UserData/sunleyao/lyc/0010/orb_0.jpg'
cv2.imwrite(output_path, orb_image)




print(f"ORB feature image has been saved to: {output_path}")
