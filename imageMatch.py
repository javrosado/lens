import cv2
import csv
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

def find_template_location(original, template):
    result = cv2.matchTemplate(original, template, cv2.TM_CCOEFF_NORMED)
    _, _, min_loc, max_loc = cv2.minMaxLoc(result)
    return max_loc



def loadSnippit(original_image, original_image_width, original_image_height):
    template_folder = input("Enter snippit folder. Remember that the patterned and unpatterned counter parts must share the same name in separate folders:")
    # Paths to the template images
    template_paths = glob.glob(str(template_folder) + "/*.png") 



    original_image_resolutionx = original_image.shape[1]
    original_image_resolutiony = original_image.shape[0]




    # Load the templates
    templates = []
    templateNames = []

    #make into case
    for template_path in template_paths: #photo size detections
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if template is None:
            raise ValueError(f"Template image {template_path} could not be loaded (╯°□°)╯︵ ┻━┻")
    #WIERD, SO IS IT .153 * 47 lenses? That is 7.2 answays
        elif template.shape[1] == 47: #7.2mm x 5.4mm
            xscale = ( (original_image_resolutionx / original_image_width) * (7.05/47))
            yscale = ( (original_image_resolutiony / original_image_height) * (5.25/35))
            print("7.2mm x 5.4mm detected ┬─┬ノ( º _ ºノ)")

        elif template.shape[1] == 35:
            xscale = ( (original_image_resolutionx / original_image_width) * (5.25/35))
            yscale = ( (original_image_resolutiony / original_image_height) * (5.25/35))
            print("5.4mm x 5.4mm detected ┬─┬ノ( º _ ºノ)")
        
        elif template.shape[1] == 23:
            xscale = ( (original_image_resolutionx / original_image_width) * (3.45/23))
            yscale = ( (original_image_resolutiony / original_image_height) * (3.45/23))
            print("3.84mm x 3.84mm detected")

        elif template.shape[1] == 15:
            xscale = ( (original_image_resolutionx / original_image_width) * (2.25/15))
            yscale = ( (original_image_resolutiony / original_image_height) * (2.25/15))
            print("2.56mm x 2.56mm detected")

        else:
            print("Size not recognized... Skipping (╯°□°)╯︵ ┻━┻")
            continue
        
       

        #properly scales
        template = cv2.resize(template, (round(template.shape[1] * yscale), round(template.shape[0] * xscale)), interpolation=cv2.INTER_CUBIC)
        templates.append(template)
        templateNames.append(os.path.basename(template_path))

    return templates, templateNames, xscale, yscale


#STUFF FOR PHASE DELAY
def mmFromCenter(original, sides, templateCenters):
   
    height, width = original.shape

    # Calculate center pixel coordinates
    center_x = width // 2
    center_y = height // 2

    # Display the center pixel coordinates
    print(f"Center Pixel Coordinates: ({center_x}, {center_y})")
    mmPerPixel = sides/width
    print(f"mm per pixel: {mmPerPixel}mm")
    print(templateCenters[0])
    mmXShift = []
    mmYShift = []

    for center in templateCenters:
        if(center[0]>= center_x):
            xshift = center[0] - center_x
        if(center[0]<center_x):
            xshift = center[0] - center_x
        if(center[1]>center_y):
            yshift = center_y - center[1]
        if(center[1]<center_y):
            yshift = center_y - center[1]
        mmXShift.append(xshift * mmPerPixel)
        mmYShift.append(yshift * mmPerPixel)
        print(mmXShift[0])
        print(mmYShift[0])
    return mmXShift, mmYShift


def makeScatterPlot(x_values, y_values, data, name, outputDir):
    plt.clf()
    plt.cla()
    X, Y = np.meshgrid(x_values, y_values)
    Z = data

    plt.scatter(X.flatten(), Y.flatten(), c=Z, cmap='viridis', vmin=-45, vmax=45)
    plt.colorbar(label='Delay')
    plt.xlabel('X Coordinates')
    plt.ylabel('Y Coordinates')
    plt.title(f'Scatter Plot for {name}')

    plt.xlim(min(x_values), max(x_values))
    plt.ylim(min(y_values), max(y_values))

    print(max(x_values))
    print(max(y_values))

    imgOutput = os.path.join(outputDir, f"{name}.png")
    plt.savefig(imgOutput)        


def getPowerArray(folder, outputDir, xshift, yshift):
    csvFiles = glob.glob(os.path.join(folder, "*.csv"))
    counter = 0

    if not csvFiles:
        print("Oops, no files found.")
        return

    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    for file in csvFiles:
        print("RUns")
        base_name = os.path.basename(file)
        name, ext = os.path.splitext(base_name)

        with open(file, 'r') as csvfile:
            lines = csvfile.readlines()

        # Find the starting line after "Spots Y"
        start_line = None
        for i, line in enumerate(lines):
            if line.strip().startswith("Spots Y"):
                start_line = i + 1  # Start processing from the next line
                break

        if start_line is None:
            print(f"No 'Spots Y' line found in {file}. Skipping.")
            continue
        
        # Read data from the identified start line
        csv_reader = csv.reader(lines[start_line:])
        data = list(csv_reader)

        # Process the data to extract coordinates
        x_values = [float(val.strip()) for val in data[0][1:] if val.strip()]  # All elements except the first from the first row
        y_values = [float(row[0].strip()) for row in data[1:] if row[0].strip()]  # All elements except the first from the first column
        plot_data = np.array([[float(val.strip()) if val.strip() else 0 for val in row[1:]] for row in data[1:]])

        # Remove the last row and last column from plot_data
        plot_data = plot_data[:-1, :-1]

        x_values = [x + xshift[counter] for x in x_values]

        y_values = [y + yshift[counter] for y in y_values]

        # Adjust x_values and y_values to match the plot_data's dimensions
        x_values = x_values[:plot_data.shape[1]]
        y_values = y_values[:plot_data.shape[0]]

        # Create output CSV file
        output_file_csv = os.path.join(outputDir, f'output{counter}.csv')
        np.savetxt(output_file_csv, plot_data, delimiter=',')

        # Create the scatter plot
        makeScatterPlot(x_values, y_values, plot_data, name, outputDir)

        counter += 1

    print(f".csv files and images saved to {outputDir}")

#END OF STUFF FOR PHASE DELAY


def powerMosiac(locations, original_image, original_image_width, original_image_height ):
    newtemplates,_,__,_ =  loadSnippit(original_image, original_image_width, original_image_height)
    data = zip(newtemplates , locations)
    canvas = np.zeros_like(original_image)
    for img, corner in data:
        x,y = corner
        h,w = img.shape

        canvas[y:y+h, x:x+w] = img
        
    cv2.imwrite('path_to_save_final_image.jpg', canvas)

def loadPixels(original_image):
    template_folder = input("Enter centroid folder. They need the same name as their patterned counterpart:")
    template_paths = glob.glob(str(template_folder) + "/*.png") 
    original_image_width = 17 #mm
    original_image_resolutionx = original_image.shape[1]
    original_image_resolutiony = original_image.shape[0]

    templates = []
    templateNames = []

    for template_path in template_paths: #photo size detections
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if template is None:
            raise ValueError(f"Template image {template_path} could not be loaded.")
            
        if template.shape[1] == 1440:
            xscale = ( 1.0 / ((original_image_resolutionx / original_image_width) * (7.2/1440.0)))
            yscale = ( 1.0 / ((original_image_resolutiony / original_image_width) * (5.4/1080.0)))
            templates.append(template)

        if template.shape[1] == 1440:

            xscale = ( 1.0 / ((original_image_resolutionx / original_image_width) * (7.2/1440.0)))
            yscale = ( 1.0 / ((original_image_resolutiony / original_image_width) * (5.4/1080.0)))
            templates.append(template)

        if template.shape[1] == 1440:

            xscale = ( 1.0 / ((original_image_resolutionx / original_image_width) * (7.2/1440.0)))
            yscale = ( 1.0 / ((original_image_resolutiony / original_image_width) * (5.4/1080.0)))
            templates.append(template)

        if template.shape[1] == 1440:

            xscale = ( 1.0 / ((original_image_resolutionx / original_image_width) * (7.2/1440.0)))
            yscale = ( 1.0 / ((original_image_resolutiony / original_image_width) * (5.4/1080.0)))
            templates.append(template)

    return templates, templateNames, xscale, yscale




def centroid(locations, original_image):
    new_templates, _, xscale, yscale = loadPixels(original_image)
    scaled_locations = [(round( x * xscale), round(y * yscale)) for x, y in locations]
    data = zip(new_templates, scaled_locations)
    canvas = np.zeros((round(original_image.shape[0] * xscale), round(original_image.shape[1] * yscale)), dtype=original_image.dtype)
    for img, corner in data:
        x,y = corner
        h,w = img.shape
        canvas[y:y+h, x:x+w] = img

    cv2.imwrite('centroidMosaic.jpg', canvas)

def phase(xshift, yshift):
    folderPath = input("Enter the directory containing .csv files:")
    outputPath = input("Enter the output directory for images: ")
    getPowerArray(folderPath, outputPath, xshift, yshift)

def ImageMatch():
    original_image_path = input("Enter pattern path")
    original_image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)

    patternWidth = int(input("Enter width of pattern in mm"))
    patternLength =int(input("Enter length of pattern in mm "))


    output_folder = input("Enter the output folder: ")
    output_locations = os.path.join(output_folder, "locationslist.txt")


    if original_image is None:
        raise ValueError(f"No image found at {original_image_path}")
        exit()

    templates, templateNames, xscale, yscale = loadSnippit(original_image,patternWidth, patternLength)







    locations = []
    loacationStrings = []
    centers = []

    #can do x plus half, y plus half. that gives center.
    for i, template in enumerate(templates):
        location = find_template_location(original_image, template)
        locations.append(location)
        print(location)
        print( template.shape[1])
        print(template.shape[0])
        centers.append((location[0] + template.shape[1]/2, location[1] + template.shape[0]/2 ))
        locationString = f"template {templateNames[i]} found with center {centers[i]} "

        print(locationString)
        loacationStrings.append(locationString)

    with open(output_locations, 'w', newline='') as file:
        for i, string in enumerate(loacationStrings):
            file.write(f"{string}\n") 

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

    answer = input("Create Mosaic :D (Y/N)? ")

    if answer == 'Y':
        powerMosiac(locations, original_image, patternWidth, patternLength)
    
    answer = input("Centroid locations :D (Y/N)?")

    if answer == 'Y':
        centroid(locations, original_image)
    
    answer = input(" Phase Delay Scatter (Y/N)?")
    if answer == 'Y':
        xshift, yshift = mmFromCenter(original_image, patternWidth, centers)
        phase(xshift, yshift)



ImageMatch()







    

