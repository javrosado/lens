import cv2
import os
import glob
import numpy as np

# Load the original image
original_image_path = input("Enter pattern path: ")
original_image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)

if original_image is None:
    raise ValueError(f"No image found at {original_image_path}")
    exit()

template_folder = input("Enter snippit folder: ")
# Paths to the template images
template_paths = glob.glob(os.path.join(template_folder, "*.png"))

output_folder = input("Enter the output folder: ")
output_locations = os.path.join(output_folder, "locationslist.txt")

original_image_width = 17  # mm
original_image_resolution = original_image.shape[1]

print(original_image_resolution)

# Scale factors
xscale = int((original_image_resolution / original_image_width) * (3.84 / 23))
yscale = int((original_image_resolution / original_image_width) * (3.84 / 23))

# Load the templates
templates = []
templateNames = []

for template_path in template_paths:
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if template is None:
        raise ValueError(f"Template image {template_path} could not be loaded.")
    template = cv2.resize(template, (template.shape[1] * yscale, template.shape[0] * xscale), interpolation=cv2.INTER_NEAREST)
    templates.append(template)
    templateNames.append(os.path.basename(template_path))

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and descriptors for the original image
keypoints1, descriptors1 = sift.detectAndCompute(original_image, None)

# Function to find the location of each template in the original image
def find_template_location(original, template):
    keypoints2, descriptors2 = sift.detectAndCompute(template, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    
    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    
    # Find homography if we have enough good matches
    if len(good_matches) > 10:
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        h, w = template.shape
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        center = np.mean(dst, axis=0).flatten()
        return (int(center[0]), int(center[1])), dst, good_matches, keypoints2
    else:
        return None, None, [], []

# Find and print the locations of all templates in the original image
locations = []
locationStrings = []

for i, template in enumerate(templates):
    location, corners, good_matches, keypoints2 = find_template_location(original_image, template)
    if location is not None:
        locations.append((location, corners, good_matches, keypoints2))
        locationString = f"template {templateNames[i]} found with center ({location[0]}, {location[1]})"
        print(locationString)
        locationStrings.append(locationString)
    else:
        print(f"template {templateNames[i]} not found.")
        locationStrings.append(f"template {templateNames[i]} not found.")

with open(output_locations, 'w', newline='') as file:
    for string in locationStrings:
        file.write(f"{string}\n")

# Visualize the locations by drawing rectangles on the original image
output_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
for location, corners, good_matches, keypoints2 in locations:
    if corners is not None:
        cv2.polylines(output_image, [np.int32(corners)], True, (0, 255, 0), 2)

# Save the output image with rectangles
output_path = os.path.join(output_folder, "marked.jpg")
cv2.imwrite(output_path, output_image)
print(f"Output image with marked locations saved at {output_path}")

# Create a combined image showing the feature matches
for i, (location, corners, good_matches, keypoints2) in enumerate(locations):
    if good_matches:
        img_matches = cv2.drawMatches(original_image, keypoints1, templates[i], keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        match_output_path = os.path.join(output_folder, f"feature_match_{templateNames[i]}.jpg")
        cv2.imwrite(match_output_path, img_matches)
        print(f"Feature matching image saved at {match_output_path}")