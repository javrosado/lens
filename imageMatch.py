import cv2
import os
import glob
import numpy as np


# Load the original image
original_image_path = input("Enter pattern path")
original_image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)



def loadSnippit():
    template_folder = input("Enter snippit folder. Remember that the patterned and unpatterned counter parts must share the same name in separate folders:")
    # Paths to the template images
    template_paths = glob.glob(str(template_folder) + "/*.png") 

    output_folder = input("Enter the output folder")
    output_locations = os.path.join(output_folder, "locationslist.txt")


    original_image_width = 17 #mm
    original_image_resolution = original_image.shape[1]




    # Load the templates
    templates = []
    templateNames = []

    #make into case
    for template_path in template_paths: #photo size detections
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if template is None:
            raise ValueError(f"Template image {template_path} could not be loaded.")
    
        if template.shape[1] == 48: #7.2mm x 5.4mm
            xscale = int( (original_image_resolution / original_image_width) * (7.2/47))
            yscale = int( (original_image_resolution / original_image_width) * (5.4/35))
            print("7.2mm x 5.4mm detected")

        if template.shape[1] == 36:
            xscale = int( (original_image_resolution / original_image_width) * (5.4/35))
            yscale = int( (original_image_resolution / original_image_width) * (5.4/35))
            print("5.4mm x 5.4mm detected")
        
        if template.shape[1] == 24:
            xscale = int( (original_image_resolution / original_image_width) * (3.84/23))
            yscale = int( (original_image_resolution / original_image_width) * (3.84/23))
            print("3.84mm x 3.84mm detected")

        if template.shape[1] == 16:
            xscale = int( (original_image_resolution / original_image_width) * (2.56/15))
            yscale = int( (original_image_resolution / original_image_width) * (2.56/15))
            print("2.56mm x 2.56mm detected")

        if template.shape[1] == 12:
            xscale = int( (original_image_resolution / original_image_width) * (1.8/1))
            yscale = int( (original_image_resolution / original_image_width) * (1.8/11))
            print("1.8mm x 1.8mm detected")

        #properly scales
        template = cv2.resize(template, (template.shape[1] * yscale, template.shape[0] * xscale), interpolation=cv2.INTER_CUBIC)
        templates.append(template)
        templateNames.append(os.path.basename(template_path))

    return templates, templateNames, output_locations





if original_image is None:
    raise ValueError(f"No image found at {original_image_path}")
    exit()

templates, templateNames, output_locations = loadSnippit()


# Function to find the location of each template in the original image
def find_template_location(original, template):
    result = cv2.matchTemplate(original, template, cv2.TM_CCOEFF_NORMED)
    _, _, min_loc, max_loc = cv2.minMaxLoc(result)
    return max_loc

# Find and print the locations of all templates in the original image
locations = []
loacationStrings = []








#can do x plus half, y plus half. that gives center.
for i, template in enumerate(templates):
    location = find_template_location(original_image, template)
    locations.append(location)
   
    locationString = f"template {templateNames[i]} found with center ({location[1] + template.shape[1]/2} , {location[0] + template.shape[0]/2}) "
    print(locationString)
    loacationStrings.append(locationString)

with open(output_locations, 'w', newline='') as file:
    for i, string in enumerate(loacationStrings):
           file.write(f"{string}\n") #WHY DOES IT NOT UPDATE THESAME 



# Visualize the locations by drawing rectangles on the original image
output_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
for i, template in enumerate(templates):
    top_left = locations[i]
    
    bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])
    cv2.rectangle(output_image, top_left, bottom_right, (0, 255, 0), 2)

# Save the output image with rectangles
output_path = "./output/marked.jpg"
cv2.imwrite(output_path, output_image)

print(f"Output image with marked locations saved at {output_path}")







#IF YOU DO THIS, THE IMAGES MUST BE IN THE SAME ORDER AS THE PATTERNED

answer = input("Create Mosaic :D (Y/N)?")

if answer == 'Y':
    newtemplates,_,_ =  loadSnippit()
    data = zip(newtemplates , locations)
    canvas = np.zeros_like(original_image)
    for img, corner in data:
        x,y = corner
        h,w = img.shape

        canvas[y:y+h, x:x+w] = img
        
    cv2.imwrite('path_to_save_final_image.jpg', canvas)

answer = input("Create Another? :D (Y/N)?")

if answer == 'Y':
    newtemplates,_,_ =  loadSnippit()
    data = zip(newtemplates , locations)
    canvas = np.zeros_like(original_image)
    for img, corner in data:
        x,y = corner
        h,w = img.shape

        canvas[y:y+h, x:x+w] = img
        
    cv2.imwrite('path_to_save_final_image2.jpg', canvas)

    

