import cv2

def sift(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Initialize the SIFT detector
    sift = cv2.SIFT_create()
    
    # Find the keypoints and descriptors with SIFT
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    
    # Draw keypoints on the image without circles
    sift_image = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0), flags=0)
    
    return sift_image

# Read the image
image_path = '/home/B_UserData/sunleyao/lyc/0010/5.jpg'
image = cv2.imread(image_path)

# Extract SIFT features
sift_image = sift(image)

# Save the image to a specified path
output_path = '/home/B_UserData/sunleyao/lyc/0010/5_sift.jpg'
cv2.imwrite(output_path, sift_image)

print(f"SIFT feature image has been saved to: {output_path}")
