import cv2
import os

# Load the original image
original_image_path = './pattern/this.jpg'
original_image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)

if original_image is None:
    raise ValueError("The original image could not be loaded. Please check the file path and file format.")

# Define the location and size of the template
# Example: location (x, y) and size (width, height)
location = (500,700)
template_width = 600
template_height = 300

# Extract the template from the original image
template = original_image[location[1]:location[1]+template_height, location[0]:location[0]+template_width]

# Save the template image
template_path = 'C:\\Users\\woode\\Desktop\\py\\template_from_location.jpg'
cv2.imwrite(template_path, template)

print(f"Template image saved to {template_path}")